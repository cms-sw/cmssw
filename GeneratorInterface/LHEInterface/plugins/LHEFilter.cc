#include <iostream>
#include <string>
#include <memory>

#include "HepMC/GenEvent.h"
#include "HepMC/SimpleVector.h"

#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

class LHEFilter : public edm::one::EDFilter<> {
public:
  explicit LHEFilter(const edm::ParameterSet &params);
  ~LHEFilter() override = default;

protected:
  bool filter(edm::Event &event, const edm::EventSetup &es) override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> hepMCToken_;
};

LHEFilter::LHEFilter(const edm::ParameterSet &params)
    : hepMCToken_(consumes<edm::HepMCProduct>(params.getParameter<edm::InputTag>("src"))) {}

bool LHEFilter::filter(edm::Event &event, const edm::EventSetup &es) {
  const edm::Handle<edm::HepMCProduct> &product = event.getHandle(hepMCToken_);

  return product->GetEvent();
}

DEFINE_FWK_MODULE(LHEFilter);
