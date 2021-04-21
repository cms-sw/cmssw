#include "GeneratorInterface/GenFilters/interface/PythiaFilterMultiAncestor.h"
#include "GeneratorInterface/GenFilters/interface/MCFilterZboostHelper.h"

#include <iostream>

using namespace edm;
using namespace std;

PythiaFilterMultiAncestor::PythiaFilterMultiAncestor(const edm::ParameterSet& iConfig)
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
      status(iConfig.getUntrackedParameter("Status", 0)),
      motherIDs(iConfig.getUntrackedParameter("MotherIDs", std::vector<int>{0})),
      daughterIDs(iConfig.getUntrackedParameter("DaughterIDs", std::vector<int>{0})),
      daughterMinPts(iConfig.getUntrackedParameter("DaughterMinPts", std::vector<double>{0.})),
      daughterMaxPts(iConfig.getUntrackedParameter("DaughterMaxPts", std::vector<double>{10000.})),
      daughterMinEtas(iConfig.getUntrackedParameter("DaughterMinEtas", std::vector<double>{-10.})),
      daughterMaxEtas(iConfig.getUntrackedParameter("DaughterMaxEtas", std::vector<double>{10.})),
      processID(iConfig.getUntrackedParameter("ProcessID", 0)),
      betaBoost(iConfig.getUntrackedParameter("BetaBoost", 0.)) {
  //now do what ever initialization is needed
}

PythiaFilterMultiAncestor::~PythiaFilterMultiAncestor() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ access the full genealogy ---------
bool PythiaFilterMultiAncestor::isAncestor(HepMC::GenParticle* particle, int IDtoMatch) {
  for (HepMC::GenVertex::particle_iterator ancestor = particle->production_vertex()->particles_begin(HepMC::ancestors);
       ancestor != particle->production_vertex()->particles_end(HepMC::ancestors);
       ++ancestor) {
    // std::cout << __LINE__ << "]\t particle's PDG ID " << particle->pdg_id()
    //                       << " \t particle's ancestor's PDG ID " << (*ancestor)->pdg_id()
    //                       << " \t ID to match " << IDtoMatch << std::endl;

    if (abs((*ancestor)->pdg_id()) == abs(IDtoMatch)) {
      //  std::cout << __LINE__ << "]\t found!" << std::endl;
      return true;
    }
  }

  // std::cout << __LINE__ << "]\t nope, no luck" << std::endl;
  return false;
}

// ------------ method called to produce the data  ------------
bool PythiaFilterMultiAncestor::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  bool accepted = false;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  if (processID == 0 || processID == myGenEvent->signal_process_id()) {
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p) {
      HepMC::FourVector mom = MCFilterZboostHelper::zboost((*p)->momentum(), betaBoost);
      rapidity = 0.5 * log((mom.e() + mom.pz()) / (mom.e() - mom.pz()));

      if (abs((*p)->pdg_id()) == particleID && mom.rho() > minpcut && mom.rho() < maxpcut &&
          (*p)->momentum().perp() > minptcut && (*p)->momentum().perp() < maxptcut && mom.eta() > minetacut &&
          mom.eta() < maxetacut && rapidity > minrapcut && rapidity < maxrapcut && (*p)->momentum().phi() > minphicut &&
          (*p)->momentum().phi() < maxphicut) {
        // find the mother
        for (std::vector<int>::const_iterator motherID = motherIDs.begin(); motherID != motherIDs.end(); ++motherID) {
          // check status if no mother's pdgID is specified
          if (status == 0 && *motherID == 0) {
            accepted = true;
          }
          if (status != 0 && *motherID == 0) {
            if ((*p)->status() == status)
              accepted = true;
          }

          // HepMC::GenParticle* mother = (*((*p)->production_vertex()->particles_in_const_begin()));

          // check the mother's pdgID
          if (status == 0 && *motherID != 0) {
            // if (abs(mother->pdg_id()) == abs(*motherID)) {
            if (isAncestor(*p, *motherID)) {
              accepted = true;
            }
          }
          if (status != 0 && *motherID != 0) {
            // if ((*p)->status() == status && abs(mother->pdg_id()) == abs(*motherID)){
            if ((*p)->status() == status && isAncestor(*p, *motherID)) {
              accepted = true;
            }
          }
        }

        // find the daughters
        if (accepted & (!daughterIDs.empty())) {
          // if you got this far it means that the mother was found
          // now let's check the daughters
          // use a counter, if there's enough daughter  that match the pdg and kinematic
          // criteria accept the event
          uint good_dau = 0;
          uint good_dau_cc = 0;
          for (HepMC::GenVertex::particle_iterator dau = (*p)->end_vertex()->particles_begin(HepMC::children);
               dau != (*p)->end_vertex()->particles_end(HepMC::children);
               ++dau) {
            for (unsigned int i = 0; i < daughterIDs.size(); ++i) {
              // if a daughter has its pdgID among the desired ones, apply kin cuts on it
              // if it survives, add a notch to the counter
              if ((*dau)->pdg_id() == daughterIDs[i]) {
                if ((*dau)->momentum().perp() < daughterMinPts[i])
                  continue;
                if ((*dau)->momentum().perp() > daughterMaxPts[i])
                  continue;
                if ((*dau)->momentum().eta() < daughterMinEtas[i])
                  continue;
                if ((*dau)->momentum().eta() > daughterMaxEtas[i])
                  continue;
                ++good_dau;
              }
              // check charge conjugation
              if (-(*dau)->pdg_id() == daughterIDs[i]) { // notice minus sign
                if ((*dau)->momentum().perp() < daughterMinPts[i])
                  continue;
                if ((*dau)->momentum().perp() > daughterMaxPts[i])
                  continue;
                if ((*dau)->momentum().eta() < daughterMinEtas[i])
                  continue;
                if ((*dau)->momentum().eta() > daughterMaxEtas[i])
                  continue;
                ++good_dau_cc;
              }
            }
          }
          if (good_dau < daughterIDs.size() && good_dau_cc < daughterIDs.size())
            accepted = false;
        }

        /*
	  if (status == 0 && motherID != 0){
	if (abs(((*p)->mother())->pdg_id()) == abs(motherID)) {
	accepted = true;
      }
      }
	if (status != 0 && motherID != 0){
	if ((*p)->status() == status && abs(((*p)->mother())->pdg_id()) == abs(motherID)){
	accepted = true;
      }
      }
         */
      }
      // only need to satisfy the conditions _once_
      if (accepted)
        break;
    }

  } else {
    accepted = true;
  }

  if (accepted) {
    return true;
  } else {
    return false;
  }
}
