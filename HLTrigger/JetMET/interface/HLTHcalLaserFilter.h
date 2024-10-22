#ifndef HLTHcalLaserFilter_h
#define HLTHcalLaserFilter_h

/** \class HLTHcalLaserFilter
 *
 *  \author Alex Mott (Caltech), Jeff Temple (FNAL)
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoMET/METAlgorithms/interface/HcalNoiseAlgo.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTHcalLaserFilter : public edm::global::EDFilter<> {
public:
  explicit HLTHcalLaserFilter(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<HcalCalibDigiCollection> m_theCalibToken;
  // parameters
  const std::vector<int> timeSlices_;
  const std::vector<double> thresholdsfC_;
  const std::vector<int> CalibCountFilterValues_;
  const std::vector<double> CalibChargeFilterValues_;
  const double maxTotalCalibCharge_;
  const int maxAllowedHFcalib_;
};

#endif  //HLTHcalLaserFilter_h
