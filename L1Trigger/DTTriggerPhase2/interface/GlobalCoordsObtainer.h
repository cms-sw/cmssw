#ifndef L1Trigger_DTTriggerPhase2_GlobalCoordsObtainer_h
#define L1Trigger_DTTriggerPhase2_GlobalCoordsObtainer_h

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

#include <cmath>
#include <fstream>
#include <iostream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

struct lut_value {
  long int a;
  long int b;
};

struct lut_group {
  std::map<int, lut_value> phic;
  std::map<int, lut_value> phi1;
  std::map<int, lut_value> phi3;
  std::map<int, lut_value> phib;
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

class GlobalCoordsObtainer {
public:
  GlobalCoordsObtainer(const edm::ParameterSet& pset);
  ~GlobalCoordsObtainer();

  void generate_luts();
  std::vector<double> get_global_coordinates(uint32_t, int, int, int);

private:
  std::map<int, lut_value> calc_atan_lut(int, int, double, double, double, int, int, int, int, int);
  // utilities to go to and from 2 complement
  int to_two_comp(int val, int size) {
    if (val >= 0)
      return val;
    return std::pow(2, size) + val;
  }

  int from_two_comp(int val, int size) { return val - ((2 * val) & (1 << size)); }

  // attributes
  bool cmssw_for_global_;
  edm::FileInPath global_coords_filename_;
  std::vector<global_constant> global_constants;
  std::map<uint32_t, lut_group> luts;
};

#endif
