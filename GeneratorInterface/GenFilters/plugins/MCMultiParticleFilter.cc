// -*- C++ -*-
//
// Package:    MCMultiParticleFilter
// Class:      MCMultiParticleFilter
//
/*

 Description: Filter to select events with an arbitrary number of given particle(s).

 Implementation: derived from MCSingleParticleFilter

*/
//
// Original Author:  Paul Lujan
//         Created:  Wed Feb 29 04:22:16 CST 2012
//
//

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <cstdlib>
#include <vector>

//
// class declaration
//

class MCMultiParticleFilter : public edm::global::EDFilter<> {
public:
  explicit MCMultiParticleFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const int numRequired_;              // number of particles required to pass filter
  const bool acceptMore_;              // if true (default), accept numRequired or more.
                                       // if false, accept events with exactly equal to numRequired.
  const std::vector<int> particleID_;  // vector of particle IDs to look for
  // the four next variables can either be a vector of length 1 (in which case the same
  // value is used for all particle IDs) or of length equal to the length of ParticleID (in which
  // case the corresponding value is used for each).
  std::vector<int> motherID_;   // mother ID of particles (optional)
  std::vector<double> ptMin_;   // minimum Pt of particles
  std::vector<double> etaMax_;  // maximum fabs(eta) of particles
  std::vector<int> status_;     // status of particles
};

MCMultiParticleFilter::MCMultiParticleFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("generator", "unsmeared")))),
      numRequired_(iConfig.getParameter<int>("NumRequired")),
      acceptMore_(iConfig.getParameter<bool>("AcceptMore")),
      particleID_(iConfig.getParameter<std::vector<int> >("ParticleID")),
      ptMin_(iConfig.getParameter<std::vector<double> >("PtMin")),
      etaMax_(iConfig.getParameter<std::vector<double> >("EtaMax")),
      status_(iConfig.getParameter<std::vector<int> >("Status")) {
  //here do whatever other initialization is needed

  // default pt, eta, status cuts to "don't care"
  std::vector<double> defptmin(1, 0);
  std::vector<double> defetamax(1, 999.0);
  std::vector<int> defstat(1, 0);
  std::vector<int> defmother;
  defmother.push_back(0);
  motherID_ = iConfig.getUntrackedParameter<std::vector<int> >("MotherID", defstat);

  // check for same size
  if ((ptMin_.size() > 1 && particleID_.size() != ptMin_.size()) ||
      (etaMax_.size() > 1 && particleID_.size() != etaMax_.size()) ||
      (status_.size() > 1 && particleID_.size() != status_.size()) ||
      (motherID_.size() > 1 && particleID_.size() != motherID_.size())) {
    edm::LogWarning("MCMultiParticleFilter") << "WARNING: MCMultiParticleFilter: size of PtMin, EtaMax, motherID, "
                                                "and/or Status does not match ParticleID size!"
                                             << std::endl;
  }

  // Fill arrays with defaults if necessary
  while (ptMin_.size() < particleID_.size())
    ptMin_.push_back(defptmin[0]);
  while (etaMax_.size() < particleID_.size())
    etaMax_.push_back(defetamax[0]);
  while (status_.size() < particleID_.size())
    status_.push_back(defstat[0]);
  while (motherID_.size() < particleID_.size())
    motherID_.push_back(defmother[0]);
}

// ------------ method called to skim the data  ------------
bool MCMultiParticleFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  int nFound = 0;

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    for (unsigned int i = 0; i < particleID_.size(); ++i) {
      if ((particleID_[i] == 0 || std::abs(particleID_[i]) == std::abs((*p)->pdg_id())) &&
          (*p)->momentum().perp() > ptMin_[i] && std::fabs((*p)->momentum().eta()) < etaMax_[i] &&
          (status_[i] == 0 || (*p)->status() == status_[i])) {
        if (motherID_[i] == 0) {  // do not check for mother ID if not sepcified
          nFound++;
          break;  // only match a given particle once!
        } else {
          bool hascorrectmother = false;
          for (HepMC::GenVertex::particles_in_const_iterator mo = (*p)->production_vertex()->particles_in_const_begin();
               mo != (*p)->production_vertex()->particles_in_const_end();
               ++mo) {
            if ((*mo)->pdg_id() == motherID_[i]) {
              hascorrectmother = true;
              break;
            }
          }
          if (hascorrectmother) {
            nFound++;
            break;  // only match a given particle once!
          }
        }
      }
    }  // loop over targets

    if (acceptMore_ && nFound == numRequired_)
      break;  // stop looking if we don't mind having more
  }           // loop over particles

  if (nFound == numRequired_) {
    return true;
  } else {
    return false;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCMultiParticleFilter);
