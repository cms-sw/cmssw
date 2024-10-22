
#include "GeneratorInterface/GenFilters/plugins/PythiaFilterMotherSister.h"
#include "GeneratorInterface/GenFilters/plugins/MCFilterZboostHelper.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;

PythiaFilterMotherSister::PythiaFilterMotherSister(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      particleID(iConfig.getUntrackedParameter("ParticleID", 0)),
      minpcut(iConfig.getUntrackedParameter("MinP", 0.)),
      maxpcut(iConfig.getUntrackedParameter("MaxP", 10000.)),
      minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
      maxptcut(iConfig.getUntrackedParameter("MaxPt", 10000.)),
      minetacut(iConfig.getUntrackedParameter("MinEta", -10.)),
      maxetacut(iConfig.getUntrackedParameter("MaxEta", 10.)),
      minrapcut(iConfig.getUntrackedParameter("MinRapidity", -20.)),
      maxrapcut(iConfig.getUntrackedParameter("MaxRapidity", 20.)),
      minphicut(iConfig.getUntrackedParameter("MinPhi", -3.5)),
      maxphicut(iConfig.getUntrackedParameter("MaxPhi", 3.5)),
      betaBoost(iConfig.getUntrackedParameter("BetaBoost", 0.)),
      motherIDs(iConfig.getUntrackedParameter("MotherIDs", std::vector<int>{0})),
      sisterID(iConfig.getUntrackedParameter("SisterID", 0)),
      maxSisDisplacement(iConfig.getUntrackedParameter("MaxSisterDisplacement", -1.)),
      nephewIDs(iConfig.getUntrackedParameter("NephewIDs", std::vector<int>{0})),
      minNephewPts(iConfig.getUntrackedParameter("MinNephewPts", std::vector<double>{0.})) {
  if (nephewIDs.size() != minNephewPts.size()) {
    throw cms::Exception("BadConfig") << "PythiaFilterMotherSister: "
                                      << "'nephewIDs' and 'minNephewPts' need same length.";
  }
}

PythiaFilterMotherSister::~PythiaFilterMotherSister() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool PythiaFilterMotherSister::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    HepMC::FourVector mom = MCFilterZboostHelper::zboost((*p)->momentum(), betaBoost);
    double rapidity = 0.5 * log((mom.e() + mom.pz()) / (mom.e() - mom.pz()));

    if (abs((*p)->pdg_id()) == particleID && mom.rho() > minpcut && mom.rho() < maxpcut &&
        (*p)->momentum().perp() > minptcut && (*p)->momentum().perp() < maxptcut && mom.eta() > minetacut &&
        mom.eta() < maxetacut && rapidity > minrapcut && rapidity < maxrapcut && (*p)->momentum().phi() > minphicut &&
        (*p)->momentum().phi() < maxphicut) {
      HepMC::GenParticle* mother = (*((*p)->production_vertex()->particles_in_const_begin()));

      // check various possible mothers
      for (auto motherID : motherIDs) {
        if (abs(mother->pdg_id()) == abs(motherID)) {
          // loop over its daughters
          for (HepMC::GenVertex::particle_iterator dau = mother->end_vertex()->particles_begin(HepMC::children);
               dau != mother->end_vertex()->particles_end(HepMC::children);
               ++dau) {
            // find the daugther you're interested in
            if (abs((*dau)->pdg_id()) == abs(sisterID)) {
              int failNephewPt = 0;
              // check pt of the nephews
              for (HepMC::GenVertex::particle_iterator nephew = (*dau)->end_vertex()->particles_begin(HepMC::children);
                   nephew != (*dau)->end_vertex()->particles_end(HepMC::children);
                   ++nephew) {
                int nephew_pdgId = abs((*nephew)->pdg_id());
                for (unsigned int i = 0; i < nephewIDs.size(); i++) {
                  if (nephew_pdgId == abs(nephewIDs.at(i)))
                    failNephewPt += ((*nephew)->momentum().perp() < minNephewPts.at(i));
                }
              }
              if (failNephewPt > 0)
                return false;
              // calculate displacement of the sister particle, from production to decay
              HepMC::GenVertex* v1 = (*dau)->production_vertex();
              HepMC::GenVertex* v2 = (*dau)->end_vertex();

              double lx12 = v1->position().x() - v2->position().x();
              double ly12 = v1->position().y() - v2->position().y();
              double lxy12 = sqrt(lx12 * lx12 + ly12 * ly12);

              if (maxSisDisplacement != -1) {
                if (lxy12 < maxSisDisplacement) {
                  return true;
                }
              } else {
                return true;
              }
            }
          }
        }
      }
    }
  }

  return false;
}
