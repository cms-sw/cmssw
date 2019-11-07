
#include "GeneratorInterface/GenFilters/plugins/PythiaProbeFilter.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;
using namespace Pythia8;

PythiaProbeFilter::PythiaProbeFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      particleID(iConfig.getUntrackedParameter("ParticleID", 0)),
      MomID(iConfig.getUntrackedParameter("MomID", 0)),
      GrandMomID(iConfig.getUntrackedParameter("GrandMomID", 0)),
      chargeconju(iConfig.getUntrackedParameter("ChargeConjugation", true)),
      nsisters(iConfig.getUntrackedParameter("NumberOfSisters", 0)),
      naunts(iConfig.getUntrackedParameter("NumberOfAunts", 0)),
      minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
      maxptcut(iConfig.getUntrackedParameter("MaxPt", 14000.)),
      minetacut(iConfig.getUntrackedParameter("MinEta", -10.)),
      maxetacut(iConfig.getUntrackedParameter("MaxEta", 10.)),
      countQEDCorPhotons(iConfig.getUntrackedParameter("countQEDCorPhotons", false)) {
  //now do what ever initialization is needed
  vector<int> defID;
  defID.push_back(0);
  exclsisIDs = iConfig.getUntrackedParameter<vector<int> >("SisterIDs", defID);
  exclauntIDs = iConfig.getUntrackedParameter<vector<int> >("AuntIDs", defID);
  identicalParticle = false;
  for (unsigned int ilist = 0; ilist < exclsisIDs.size(); ++ilist) {
    if (fabs(exclsisIDs[ilist]) == fabs(particleID))
      identicalParticle = true;
  }
  // create pythia8 instance to access particle data
  edm::LogInfo("PythiaProbeFilter::PythiaProbeFilter") << "Creating pythia8 instance for particle properties" << endl;
  if (!fLookupGen.get())
    fLookupGen.reset(new Pythia());
}

PythiaProbeFilter::~PythiaProbeFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

bool PythiaProbeFilter::AlreadyExcludedCheck(std::vector<unsigned int> excludedList, unsigned int current_part) const {
  bool result = false;
  for (unsigned int checkNow : excludedList) {
    if (current_part != checkNow)
      continue;
    result = true;
    break;
  }
  return result;
}
//
// member functions
//
// ------------ method called to produce the data  ------------
bool PythiaProbeFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  bool accepted = false;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  //access particles
  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    //select tag particle
    if (fabs((*p)->pdg_id()) == fabs(particleID))
      if ((*p)->pdg_id() != particleID && !chargeconju)
        continue;

    if (fabs((*p)->pdg_id()) != fabs(particleID) && chargeconju)
      continue;
    //kinematic properties of tag
    if ((*p)->momentum().perp() < minptcut || (*p)->momentum().perp() > maxptcut)
      continue;
    if ((*p)->momentum().eta() < minetacut || (*p)->momentum().eta() > maxetacut)
      continue;
    //remove probe side particles
    bool excludeTagParticle = false;
    if (naunts == 0) {
      if ((*p)->production_vertex()) {
        for (HepMC::GenVertex::particle_iterator anc = (*p)->production_vertex()->particles_begin(HepMC::parents);
             anc != (*p)->production_vertex()->particles_end(HepMC::parents);
             ++anc) {
          if (fabs((*anc)->pdg_id()) != fabs(MomID) && chargeconju)
            continue;
          else if ((*anc)->pdg_id() != MomID && !chargeconju)
            continue;
          int nsis = 0;
          int exclsis = 0;
          std::vector<unsigned int> checklistSis;
          if ((*anc)->end_vertex()) {
            for (HepMC::GenVertex::particle_iterator sis = (*anc)->end_vertex()->particles_begin(HepMC::children);
                 sis != (*anc)->end_vertex()->particles_end(HepMC::children);
                 ++sis) {
              //identify the tag particle in the decay
              if ((*p)->pdg_id() == (*sis)->pdg_id() && (identicalParticle || !chargeconju))
                continue;
              if (fabs((*p)->pdg_id()) == fabs((*sis)->pdg_id()) && !identicalParticle && chargeconju)
                continue;
              //remove QED photons
              if ((*sis)->pdg_id() == 22 && !countQEDCorPhotons)
                continue;
              nsis++;
              //check if this sis is excluded already
              for (unsigned int ilist = 0; ilist < exclsisIDs.size(); ++ilist) {
                if (AlreadyExcludedCheck(checklistSis, ilist)) {
                  continue;
                }
                if (fabs(exclsisIDs[ilist]) == fabs((*sis)->pdg_id()) && chargeconju) {
                  exclsis++;
                  checklistSis.push_back(ilist);
                }
                if (exclsisIDs[ilist] == (*sis)->pdg_id() && !chargeconju) {
                  exclsis++;
                  checklistSis.push_back(ilist);
                }
              }
            }
          }
          if (nsis == exclsis && nsis == nsisters) {
            excludeTagParticle = true;
            break;
          }
        }
      }
    } else if (naunts > 0) {
      //now take into account that we have up 2 generations in the decay
      if ((*p)->production_vertex()) {
        for (HepMC::GenVertex::particle_iterator anc = (*p)->production_vertex()->particles_begin(HepMC::parents);
             anc != (*p)->production_vertex()->particles_end(HepMC::parents);
             ++anc) {
          if (fabs((*anc)->pdg_id()) != fabs(MomID) && chargeconju)
            continue;
          else if ((*anc)->pdg_id() != MomID && !chargeconju)
            continue;
          int nsis = 0;
          int exclsis = 0;
          std::vector<unsigned int> checklistSis;
          int naunt = 0;
          int exclaunt = 0;
          std::vector<unsigned int> checklistAunt;
          if ((*anc)->end_vertex()) {
            for (HepMC::GenVertex::particle_iterator sis = (*anc)->end_vertex()->particles_begin(HepMC::children);
                 sis != (*anc)->end_vertex()->particles_end(HepMC::children);
                 ++sis) {
              //identify the particle under study in the decay
              if ((*p)->pdg_id() == (*sis)->pdg_id() && (identicalParticle || !chargeconju))
                continue;
              if (fabs((*p)->pdg_id()) == fabs((*sis)->pdg_id()) && !identicalParticle && chargeconju)
                continue;
              //remove QED photons
              if ((*sis)->pdg_id() == 22 && !countQEDCorPhotons)
                continue;
              nsis++;
              for (unsigned int ilist = 0; ilist < exclsisIDs.size(); ++ilist) {
                if (AlreadyExcludedCheck(checklistSis, ilist))
                  continue;
                if (fabs(exclsisIDs[ilist]) == fabs((*sis)->pdg_id()) && chargeconju) {
                  exclsis++;
                  checklistSis.push_back(ilist);
                }
                if (exclsisIDs[ilist] == (*sis)->pdg_id() && !chargeconju) {
                  exclsis++;
                  checklistSis.push_back(ilist);
                }
              }
            }
          }
          //check sisters
          if (nsis != exclsis || nsis != nsisters)
            break;
          if ((*anc)->production_vertex()) {
            for (HepMC::GenVertex::particle_iterator granc =
                     (*anc)->production_vertex()->particles_begin(HepMC::parents);
                 granc != (*anc)->production_vertex()->particles_end(HepMC::parents);
                 ++granc) {
              if (fabs((*granc)->pdg_id()) != fabs(GrandMomID) && chargeconju)
                continue;
              else if ((*granc)->pdg_id() != GrandMomID && !chargeconju)
                continue;
              for (HepMC::GenVertex::particle_iterator aunt = (*granc)->end_vertex()->particles_begin(HepMC::children);
                   aunt != (*granc)->end_vertex()->particles_end(HepMC::children);
                   ++aunt) {
                if ((*aunt)->pdg_id() == (*anc)->pdg_id())
                  continue;
                if ((*aunt)->pdg_id() == 22 && !countQEDCorPhotons)
                  continue;
                naunt++;
                for (unsigned int ilist = 0; ilist < exclauntIDs.size(); ++ilist) {
                  if (AlreadyExcludedCheck(checklistAunt, ilist))
                    continue;
                  if (fabs(exclauntIDs[ilist]) == fabs((*aunt)->pdg_id()) && chargeconju) {
                    exclaunt++;
                    checklistAunt.push_back(ilist);
                  }
                  if (exclauntIDs[ilist] == (*aunt)->pdg_id() && !chargeconju) {
                    exclaunt++;
                    checklistAunt.push_back(ilist);
                  }
                }
              }
            }
          }
          //check aunts
          if (naunt == exclaunt && naunt == naunts) {
            excludeTagParticle = true;
            break;
          }
        }
      }
    }
    if (excludeTagParticle)
      continue;
    accepted = true;
    break;
  }

  if (accepted) {
    return true;
  } else {
    return false;
  }
}
