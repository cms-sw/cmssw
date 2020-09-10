// -*- C++ -*-
//
// Package:    RecoMET/METFilters
// Class:      HFNoisyHitsFilter
//
/**\class HFNoisyHitsFilter HFNoisyHitsFilter.cc RecoMET/METFilters/plugins/HFNoisyHitsFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Laurent Thomas
//         Created:  Tue, 01 Sep 2020 11:24:33 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/METReco/interface/HcalPhase1FlagLabels.h"
//
// class declaration
//

class HFNoisyHitsFilter : public edm::global::EDFilter<> {
public:
  explicit HFNoisyHitsFilter(const edm::ParameterSet&);
  ~HFNoisyHitsFilter() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  std::vector<HcalPhase1FlagLabels::HFStatusFlag> getNoiseBits() const;
  const edm::EDGetTokenT<HFRecHitCollection> hfhits_token_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geom_token_;
  const double rechitPtThreshold_;
  const std::vector<std::string> listOfNoises_;
  const bool taggingMode_;
  const bool debug_;
  std::vector<HcalPhase1FlagLabels::HFStatusFlag> noiseBits_;
};

//
// constructors and destructor
//
HFNoisyHitsFilter::HFNoisyHitsFilter(const edm::ParameterSet& iConfig)
    : hfhits_token_(consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfrechits"))),
      geom_token_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      rechitPtThreshold_(iConfig.getParameter<double>("rechitPtThreshold")),
      listOfNoises_(iConfig.getParameter<std::vector<std::string>>("listOfNoises")),
      taggingMode_(iConfig.getParameter<bool>("taggingMode")),
      debug_(iConfig.getParameter<bool>("debug")) {
  noiseBits_ = getNoiseBits();
  produces<bool>();
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HFNoisyHitsFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  bool pass = true;

  // Calo Geometry - needed for computing E_t
  const CaloGeometry& geo = iSetup.getData(geom_token_);

  auto const& hfHits = iEvent.get(hfhits_token_);

  //Loop over the HF rechits. If one of them has Et>X and fires one the noise bits, declare the event as bad
  for (auto const& hfhit : hfHits) {
    float ene = hfhit.energy();
    float et = 0;
    // compute transverse energy
    const GlobalPoint& poshf = geo.getPosition(hfhit.detid());
    float pf = poshf.perp() / poshf.mag();
    et = ene * pf;
    if (et < rechitPtThreshold_)
      continue;
    int hitFlags = hfhit.flags();
    for (auto noiseBit : noiseBits_) {
      if ((hitFlags >> noiseBit) & 1) {
        pass = false;
        break;
      }
    }
    if (!pass)
      break;
  }
  iEvent.put(std::make_unique<bool>(pass));
  if (debug_)
    LogDebug("HFNoisyHitsFilter") << "Passing filter? " << pass;
  return taggingMode_ || pass;
}

std::vector<HcalPhase1FlagLabels::HFStatusFlag> HFNoisyHitsFilter::getNoiseBits() const {
  std::vector<HcalPhase1FlagLabels::HFStatusFlag> result;
  for (auto const& noise : listOfNoises_) {
    if (noise == "HFLongShort")
      result.push_back(HcalPhase1FlagLabels::HFLongShort);
    else if (noise == "HFS8S1Ratio")
      result.push_back(HcalPhase1FlagLabels::HFS8S1Ratio);
    else if (noise == "HFPET")
      result.push_back(HcalPhase1FlagLabels::HFPET);
    else if (noise == "HFSignalAsymmetry")
      result.push_back(HcalPhase1FlagLabels::HFSignalAsymmetry);
    else if (noise == "HFAnomalousHit")
      result.push_back(HcalPhase1FlagLabels::HFAnomalousHit);
    else
      throw cms::Exception("Error") << "Couldn't find the bit index associated to this string: " << noise;
  }

  return result;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HFNoisyHitsFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hfrechits", {"reducedHcalRecHits:hfreco"});
  desc.add<double>("rechitPtThreshold", 20.);
  desc.add<std::vector<std::string>>("listOfNoises", {"HFLongShort", "HFS8S1Ratio", "HFPET", "HFSignalAsymmetry"});
  desc.add<bool>("taggingMode", false);
  desc.add<bool>("debug", false);
  descriptions.add("hfNoisyHitsFilter", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(HFNoisyHitsFilter);
