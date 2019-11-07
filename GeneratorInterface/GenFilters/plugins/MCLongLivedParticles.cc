
#include "GeneratorInterface/GenFilters/plugins/MCLongLivedParticles.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;

MCLongLivedParticles::MCLongLivedParticles(const edm::ParameterSet& iConfig)
    : hepMCProductTag_(
          iConfig.getUntrackedParameter<edm::InputTag>("hepMCProductTag", edm::InputTag("generator", "unsmeared"))) {
  //here do whatever other initialization is needed
  theCut = iConfig.getUntrackedParameter<double>("LengCut", 10.);
}

MCLongLivedParticles::~MCLongLivedParticles() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to skim the data  ------------
bool MCLongLivedParticles::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<HepMCProduct> evt;

  iEvent.getByLabel(hepMCProductTag_, evt);

  bool pass = false;

  const HepMC::GenEvent* generated_event = evt->GetEvent();
  HepMC::GenEvent::particle_const_iterator p;

  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++) {
    if (((*p)->production_vertex() != nullptr) && ((*p)->end_vertex() != nullptr)) {
      float dist = sqrt((((*p)->production_vertex())->position().x() - ((*p)->end_vertex())->position().x()) *
                            (((*p)->production_vertex())->position().x() - ((*p)->end_vertex())->position().x()) +
                        (((*p)->production_vertex())->position().y() - ((*p)->end_vertex())->position().y()) *
                            (((*p)->production_vertex())->position().y() - ((*p)->end_vertex())->position().y()));
      if (dist > theCut)
        pass = true;
    }

    if (((*p)->production_vertex() == nullptr) && (!((*p)->end_vertex() != nullptr))) {
      if (((*p)->end_vertex())->position().perp() > theCut)
        pass = true;
    }

    if (pass)
      return pass;
  }

  return pass;
}
