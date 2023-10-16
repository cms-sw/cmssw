#ifndef PhotonGenFilter_h
#define PhotonGenFilter_h

#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

namespace edm {
  class HepMCProduct;
}

class PhotonGenFilter : public edm::global::EDFilter<> {
public:
  explicit PhotonGenFilter(const edm::ParameterSet&);
  ~PhotonGenFilter() override;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  bool hasAncestor(HepMC::GenParticle* particle,
                   std::function<bool(int)> check,
                   bool isWorBFromDecayCheck = false,
                   bool isWorBPromptCheck = false) const;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // Private member variables and functions

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  double ptMin;
  double etaMin;
  double etaMax;
  double drMin;
  double ptThreshold;
};

#endif