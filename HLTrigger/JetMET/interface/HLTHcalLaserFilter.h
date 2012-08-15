#ifndef HLTHcalLaserFilter_h
#define HLTHcalLaserFilter_h

/** \class HLTHcalLaserFilter
 *
 *  \author Alex Mott (Caltech), Jeff Temple (FNAL)
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoMET/METAlgorithms/interface/HcalNoiseAlgo.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTHcalLaserFilter : public edm::EDFilter {
  
 public:
  explicit HLTHcalLaserFilter(const edm::ParameterSet&);
  ~HLTHcalLaserFilter();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
 private:
  // parameters
  edm::InputTag hcalDigiCollection_;
  double maxTotalCalibCharge_;
  int maxCalibCountTS45_;
  int maxCalibCountgt15TS45_;
  double maxCalibChargeTS45_;
  double maxCalibChargegt15TS45_;
};

#endif //HLTHcalLaserFilter_h
