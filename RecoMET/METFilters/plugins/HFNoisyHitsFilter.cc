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
  std::vector<int> getNoiseBits(std::vector<std::string> noiseList) const;
  const edm::EDGetTokenT<HFRecHitCollection> hfhits_token_;
  const double rechitPtThreshold_;
  const std::vector<std::string> listOfNoises_;
  const bool taggingMode_;
  const bool debug_;
};

//
// constructors and destructor
//
HFNoisyHitsFilter::HFNoisyHitsFilter(const edm::ParameterSet& iConfig)
    : hfhits_token_(consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfrechits"))),
      rechitPtThreshold_(iConfig.getParameter<double>("rechitPtThreshold")),
      listOfNoises_(iConfig.getParameter<std::vector<std::string>>("listOfNoises")),
      taggingMode_(iConfig.getParameter<bool>("taggingMode")),
      debug_(iConfig.getParameter<bool>("debug")) {
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
  ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();

  edm::Handle<HFRecHitCollection> theHFhits;
  iEvent.getByToken(hfhits_token_, theHFhits);

  std::vector<int> noiseBits = getNoiseBits(listOfNoises_);

  //Loop over the HF rechits. If one of them has Et>X and fires one the noise bits, declare the event as bad
  for (HFRecHitCollection::const_iterator hfhit = theHFhits->begin(); hfhit != theHFhits->end(); hfhit++) {
    float ene = hfhit->energy();
    float et = 0;
    // compute transverse energy
    const GlobalPoint& poshf = geo->getPosition(hfhit->detid());
    float pf = poshf.perp() / poshf.mag();
    et = ene * pf;
    if (et < rechitPtThreshold_)
      continue;
    int hitFlags = hfhit->flags();
    for (unsigned int i = 0; i < noiseBits.size(); i++) {
      if (noiseBits[i] < 0)
        continue;
      if (((hitFlags >> noiseBits[i]) & 1) > 0) {
        pass = false;
        break;
      }
    }
    if (!pass)
      break;
  }
  iEvent.put(std::make_unique<bool>(pass));
  return taggingMode_ || pass;
}

std::vector<int> HFNoisyHitsFilter::getNoiseBits(std::vector<std::string> noiseList) const {
  std::vector<int> result;
  for (unsigned int i = 0; i < noiseList.size(); i++) {
    if (noiseList[i] == "HFLongShort")
      result.push_back(HcalPhase1FlagLabels::HFLongShort);
    else if (noiseList[i] == "HFS8S1Ratio")
      result.push_back(HcalPhase1FlagLabels::HFS8S1Ratio);
    else if (noiseList[i] == "HFPET")
      result.push_back(HcalPhase1FlagLabels::HFPET);
    else if (noiseList[i] == "HFSignalAsymmetry")
      result.push_back(HcalPhase1FlagLabels::HFSignalAsymmetry);
    else if (noiseList[i] == "HFAnomalousHit")
      result.push_back(HcalPhase1FlagLabels::HFAnomalousHit);
    else if (debug_)
      edm::LogWarning("HFNoisyHitsFilter") << "Couldn't find the bit index associated to this string: " << noiseList[i];
  }

  return result;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HFNoisyHitsFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hfrechits", edm::InputTag("reducedHcalRecHits:hfreco"));
  desc.add<double>("rechitPtThreshold", 20.);
  desc.add<std::vector<std::string>>("listOfNoises", {"HFLongShort", "HFS8S1Ratio", "HFPET", "HFSignalAsymmetry"});
  desc.add<bool>("taggingMode", false);
  desc.add<bool>("debug", false);
  descriptions.add("hfNoisyHitsFilter", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(HFNoisyHitsFilter);
