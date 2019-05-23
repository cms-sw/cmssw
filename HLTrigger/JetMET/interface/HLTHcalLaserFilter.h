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
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTHcalLaserFilter : public edm::EDFilter {
public:
  explicit HLTHcalLaserFilter(const edm::ParameterSet&);
  ~HLTHcalLaserFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<HcalCalibDigiCollection> m_theCalibToken;
  // parameters
  edm::InputTag hcalDigiCollection_;
  std::vector<int> timeSlices_;
  std::vector<double> thresholdsfC_;
  std::vector<int> CalibCountFilterValues_;
  std::vector<double> CalibChargeFilterValues_;
  double maxTotalCalibCharge_;
  int maxAllowedHFcalib_;
};

#endif  //HLTHcalLaserFilter_h
