// description: select events where the multiplicity of specified particle is a multiple of provided number

// system include files
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

class MCParticleModuloFilter : public edm::global::EDFilter<> {
public:
  explicit MCParticleModuloFilter(const edm::ParameterSet&);
  ~MCParticleModuloFilter() override {}

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // member data
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> particleIDs_;
  unsigned multipleOf_;
  bool absID_;
  unsigned min_;
  int status_;
};

MCParticleModuloFilter::MCParticleModuloFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("moduleLabel"))),
      particleIDs_(iConfig.getParameter<std::vector<int>>("particleIDs")),
      multipleOf_(iConfig.getParameter<unsigned>("multipleOf")),
      absID_(iConfig.getParameter<bool>("absID")),
      min_(iConfig.getParameter<unsigned>("min")),
      status_(iConfig.getParameter<int>("status")) {
  // allow binary search
  std::sort(particleIDs_.begin(), particleIDs_.end());
}

bool MCParticleModuloFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // get input
  edm::Handle<edm::HepMCProduct> h_evt;
  iEvent.getByToken(token_, h_evt);
  const HepMC::GenEvent* myGenEvent = h_evt->GetEvent();

  unsigned sum_part = 0;
  for (auto i_part = myGenEvent->particles_begin(); i_part != myGenEvent->particles_end(); ++i_part) {
    int id_part = (*i_part)->pdg_id();
    if (absID_)
      id_part = std::abs(id_part);
    if (std::binary_search(particleIDs_.begin(), particleIDs_.end(), id_part) and
        (status_ == 0 or (*i_part)->status() == status_))
      ++sum_part;
  }

  return (sum_part % multipleOf_ == 0) and (sum_part >= min_);
}

void MCParticleModuloFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("moduleLabel", edm::InputTag("generator", "unsmeared"));
  desc.add<std::vector<int>>("particleIDs", {});
  desc.add<unsigned>("multipleOf", 1);
  desc.add<bool>("absID", false);
  desc.add<unsigned>("min", 0);
  desc.add<int>("status", 0);

  descriptions.add("MCParticleModuloFilter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCParticleModuloFilter);
