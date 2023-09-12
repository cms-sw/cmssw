#include "GeneratorInterface/GenFilters/plugins/PhotonGenFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "GeneratorInterface/GenFilters/plugins/MCFilterZboostHelper.h"
#include <iostream>

using namespace edm;
using namespace std;

PhotonGenFilter::PhotonGenFilter(const edm::ParameterSet &iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))) {
  // Constructor implementation
  ptMin = iConfig.getUntrackedParameter<double>("MinPt", 20.);
  etaMin = iConfig.getUntrackedParameter<double>("MinEta", -2.4);
  etaMax = iConfig.getUntrackedParameter<double>("MaxEta", 2.4);
  drMin = iConfig.getUntrackedParameter<double>("drMin", 0.1);
  ptThreshold = iConfig.getUntrackedParameter<double>("ptThreshold", 2.);
}

PhotonGenFilter::~PhotonGenFilter() {
  // Destructor implementation
}

bool PhotonGenFilter::filter(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  const HepMC::GenEvent *myGenEvent = evt->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if ((*p)->pdg_id() == 22) {
      if ((*p)->momentum().perp() > ptMin && (*p)->status() == 1 && (*p)->momentum().eta() > etaMin &&
          (*p)->momentum().eta() < etaMax) {
        bool accepted_photon = true;
        double phi = (*p)->momentum().phi();
        double eta = (*p)->momentum().eta();
        for (HepMC::GenEvent::particle_const_iterator q = myGenEvent->particles_begin();
             q != myGenEvent->particles_end();
             ++q) {
          if (&p != &q) {
            if ((*q)->momentum().perp() > ptThreshold && (*q)->pdg_id() != 22 &&
                (*q)->status() == 1)  // && abs((*q)->charge()) > 0)
            {
              double phi2 = (*p)->momentum().phi();
              double deltaphi = fabs(phi - phi2);
              if (deltaphi > M_PI)
                deltaphi = 2. * M_PI - deltaphi;
              double eta2 = (*p)->momentum().eta();
              double deltaeta = fabs(eta - eta2);
              double deltaR = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);
              if (deltaR < drMin)
                accepted_photon = false;
            }
          }
        }
        if (accepted_photon)
          return true;
      }
    }
  }

  // Implementation for event filtering
  return false;  // Return true if event passes the filter, false otherwise
}

void PhotonGenFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<double>("MaxEta", 2.4);
  desc.addUntracked<double>("MinEta", -2.4);
  desc.addUntracked<double>("MinPt", 20.);
  desc.addUntracked<double>("drMin", 0.1);
  desc.addUntracked<double>("ptThreshold", 2.);

  descriptions.add("PhotonGenFilter", desc);
}

DEFINE_FWK_MODULE(PhotonGenFilter);