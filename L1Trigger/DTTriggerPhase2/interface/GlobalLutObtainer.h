#ifndef L1Trigger_DTTriggerPhase2_GlobalLutObtainer_h
#define L1Trigger_DTTriggerPhase2_GlobalLutObtainer_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include <iostream>
#include <fstream>
#include <math.h> 


// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

struct lut_value {
  long int a;
  long int b;
};

struct lut_group {
  std::map <int, lut_value> phic;
  std::map <int, lut_value> phi1;
  std::map <int, lut_value> phi3;
  std::map <int, lut_value> phib;
};

struct global_constant_per_sl {
  double perp; 
  double x_phi0;
};

struct global_constant {
  uint32_t chid;
  global_constant_per_sl sl1;
  global_constant_per_sl sl3;
};


// ===============================================================================
// Class declarations
// ===============================================================================

class GlobalLutObtainer {
public:
  GlobalLutObtainer(const edm::ParameterSet& pset);
  ~GlobalLutObtainer();

  void generate_luts();
  lut_group get_luts(uint32_t chid) {return luts[chid];}
 
private:
  edm::FileInPath global_coords_filename_;
  std::vector<global_constant> global_constants;
  std::map<uint32_t, lut_group> luts;

  std::map<int, lut_value> calc_atan_lut(int, int, double, double, double, int, int, int, int, int);
};

#endif