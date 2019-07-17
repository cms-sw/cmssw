#include "L1Trigger/L1CaloTrigger/interface/L1EGammaEECalibrator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include<iterator>
// #include <iostream>

std::vector<float> as_vector(boost::property_tree::ptree const& pt, boost::property_tree::ptree::key_type const& key)
{
    std::vector<float> ret;
    for (auto& item : pt.get_child(key))
        ret.push_back(item.second.get_value<float>());
    return ret;
}




L1EGammaEECalibrator::L1EGammaEECalibrator(const edm::ParameterSet& pset) {
    //read the JSON file and populate the eta - pt bins and the value container
    boost::property_tree::ptree calibration_map;
    read_json(pset.getParameter<edm::FileInPath>("calirationFile").fullPath(), calibration_map);

    auto eta_l = as_vector(calibration_map, "eta_l");
    std::copy(eta_l.begin(), eta_l.end(), std::inserter(eta_bins, eta_bins.end()));
    auto eta_h = as_vector(calibration_map, "eta_h");
    eta_bins.insert(eta_h.back());

    // std::cout << "# of eta bins: " << eta_bins.size()-1 << std::endl;
    auto pt_l = as_vector(calibration_map, "pt_l");
    std::copy(pt_l.begin(), pt_l.end(), std::inserter(pt_bins, pt_bins.end()));
    auto pt_h = as_vector(calibration_map, "pt_h");
    pt_bins.insert(pt_h.back());
    // std::cout << "# of pt bins: " << pt_bins.size()-1 << std::endl;

    auto calib_data = as_vector(calibration_map, "calib");
    auto n_bins_eta = eta_bins.size();
    auto n_bins_pt = pt_bins.size();
    calib_factors.reserve(n_bins_eta*n_bins_pt);
    for (auto calib_f = calib_data.begin(); calib_f != calib_data.end(); ++calib_f) {
      auto index = calib_f - calib_data.begin();
      int eta_bin = get_eta_bin(eta_l[index]);
      int pt_bin = get_pt_bin(pt_l[index]);
      calib_factors[(eta_bin*n_bins_pt)+pt_bin] = *calib_f;
    }
}

int L1EGammaEECalibrator::get_bin(const std::set<float>& container, float value) const {
  auto bin_l = container.upper_bound(value);
  // std::cout << "value " << value << "lower boud " << *bin_l << " distance: " << std::distance(container.begin(), bin_l)-1 << std::endl;
  if(bin_l == container.end()) {
    // value not mapped to any bin
    return -1;
  }
  // return bin_l - container.begin();
  return std::distance(container.begin(), bin_l)-1;

}

float L1EGammaEECalibrator::calibrationFactor(const float& pt, const float& eta) const {
  int bin_eta = get_eta_bin(eta);
  int bin_pt = get_pt_bin(pt);
  if (bin_eta == -1 || bin_pt == -1) return 1.;
  auto n_bins_pt = pt_bins.size();
  // std::cout << "pt: " << pt << " eta: " << eta << " ptbin: " << bin_pt << " etabin: " << bin_eta << " calib: " <<  calib_factors[(bin_eta*n_bins_pt)+bin_pt] << std::endl;
  return calib_factors[(bin_eta*n_bins_pt)+bin_pt];
}
