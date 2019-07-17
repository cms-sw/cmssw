#ifndef L1Trigger_L1CaloTrigger_L1EGammaEECalibrator_h
#define L1Trigger_L1CaloTrigger_L1EGammaEECalibrator_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <vector>
#include <set>
#include <cmath>

class L1EGammaEECalibrator {
   public:
      explicit L1EGammaEECalibrator(const edm::ParameterSet&);

      float calibrationFactor(const float& pt, const float& eta) const;

   private:
     int get_eta_bin(float eta) const {
       return get_bin(eta_bins, fabs(eta)); }
     int get_pt_bin(float pt) const {
      return get_bin(pt_bins, pt); }
     int get_bin(const std::set<float>& container, float value) const;

     std::set<float> eta_bins;
     std::set<float> pt_bins;
     std::vector<float> calib_factors;
};

#endif
