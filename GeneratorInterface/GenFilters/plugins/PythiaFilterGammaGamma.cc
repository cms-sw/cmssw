#include "GeneratorInterface/GenFilters/plugins/PythiaFilterGammaGamma.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TLorentzVector.h"

#include <iostream>

using namespace edm;
using namespace std;
using namespace HepMC;

PythiaFilterGammaGamma::PythiaFilterGammaGamma(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      hepMCFilter_(new PythiaHepMCFilterGammaGamma(iConfig)) {}

PythiaFilterGammaGamma::~PythiaFilterGammaGamma() {}

bool PythiaFilterGammaGamma::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  return hepMCFilter_->filter(myGenEvent);
}
