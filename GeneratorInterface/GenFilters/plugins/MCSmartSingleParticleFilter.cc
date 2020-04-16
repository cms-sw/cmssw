// -*- C++ -*-
//
// Package:    MCSmartSingleParticleFilter
// Class:      MCSmartSingleParticleFilter
//
/*

 Description: filter events based on the Pythia particleID, the Pt and the production vertex

 Implementation: inherits from generic EDFilter

*/
//         Created:  J. Alcaraz, 04/07/2008
//

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "GeneratorInterface/GenFilters/plugins/MCFilterZboostHelper.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

class MCSmartSingleParticleFilter : public edm::global::EDFilter<> {
public:
  explicit MCSmartSingleParticleFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> particleID;
  std::vector<double> pMin;
  std::vector<double> ptMin;
  std::vector<double> etaMin;
  std::vector<double> etaMax;
  std::vector<int> status;
  std::vector<double> decayRadiusMin;
  std::vector<double> decayRadiusMax;
  std::vector<double> decayZMin;
  std::vector<double> decayZMax;
  const double betaBoost;
};

using namespace std;

MCSmartSingleParticleFilter::MCSmartSingleParticleFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      betaBoost(iConfig.getUntrackedParameter("BetaBoost", 0.)) {
  vector<int> defpid;
  defpid.push_back(0);
  particleID = iConfig.getUntrackedParameter<vector<int> >("ParticleID", defpid);
  vector<double> defpmin;
  defpmin.push_back(0.);
  pMin = iConfig.getUntrackedParameter<vector<double> >("MinP", defpmin);

  vector<double> defptmin;
  defptmin.push_back(0.);
  ptMin = iConfig.getUntrackedParameter<vector<double> >("MinPt", defptmin);

  vector<double> defetamin;
  defetamin.push_back(-10.);
  etaMin = iConfig.getUntrackedParameter<vector<double> >("MinEta", defetamin);
  vector<double> defetamax;
  defetamax.push_back(10.);
  etaMax = iConfig.getUntrackedParameter<vector<double> >("MaxEta", defetamax);
  vector<int> defstat;
  defstat.push_back(0);
  status = iConfig.getUntrackedParameter<vector<int> >("Status", defstat);

  vector<double> defDecayRadiusmin;
  defDecayRadiusmin.push_back(-1.);
  decayRadiusMin = iConfig.getUntrackedParameter<vector<double> >("MinDecayRadius", defDecayRadiusmin);

  vector<double> defDecayRadiusmax;
  defDecayRadiusmax.push_back(1.e5);
  decayRadiusMax = iConfig.getUntrackedParameter<vector<double> >("MaxDecayRadius", defDecayRadiusmax);

  vector<double> defDecayZmin;
  defDecayZmin.push_back(-1.e5);
  decayZMin = iConfig.getUntrackedParameter<vector<double> >("MinDecayZ", defDecayZmin);

  vector<double> defDecayZmax;
  defDecayZmax.push_back(1.e5);
  decayZMax = iConfig.getUntrackedParameter<vector<double> >("MaxDecayZ", defDecayZmax);

  // check for same size
  if ((pMin.size() > 1 && particleID.size() != pMin.size()) ||
      (ptMin.size() > 1 && particleID.size() != ptMin.size()) ||
      (etaMin.size() > 1 && particleID.size() != etaMin.size()) ||
      (etaMax.size() > 1 && particleID.size() != etaMax.size()) ||
      (status.size() > 1 && particleID.size() != status.size()) ||
      (decayRadiusMin.size() > 1 && particleID.size() != decayRadiusMin.size()) ||
      (decayRadiusMax.size() > 1 && particleID.size() != decayRadiusMax.size()) ||
      (decayZMin.size() > 1 && particleID.size() != decayZMin.size()) ||
      (decayZMax.size() > 1 && particleID.size() != decayZMax.size())) {
    edm::LogError("Configuration")
        << "WARNING: MCPROCESSFILTER : size of MinPthat and/or MaxPthat not matching with ProcessID size!!";
  }

  // if pMin size smaller than particleID , fill up further with defaults
  if (particleID.size() > pMin.size()) {
    vector<double> defpmin2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      defpmin2.push_back(0.);
    }
    pMin = defpmin2;
  }
  // if ptMin size smaller than particleID , fill up further with defaults
  if (particleID.size() > ptMin.size()) {
    vector<double> defptmin2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      defptmin2.push_back(0.);
    }
    ptMin = defptmin2;
  }
  // if etaMin size smaller than particleID , fill up further with defaults
  if (particleID.size() > etaMin.size()) {
    vector<double> defetamin2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      defetamin2.push_back(-10.);
    }
    etaMin = defetamin2;
  }
  // if etaMax size smaller than particleID , fill up further with defaults
  if (particleID.size() > etaMax.size()) {
    vector<double> defetamax2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      defetamax2.push_back(10.);
    }
    etaMax = defetamax2;
  }
  // if status size smaller than particleID , fill up further with defaults
  if (particleID.size() > status.size()) {
    vector<int> defstat2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      defstat2.push_back(0);
    }
    status = defstat2;
  }

  // if decayRadiusMin size smaller than particleID , fill up further with defaults
  if (particleID.size() > decayRadiusMin.size()) {
    vector<double> decayRadiusmin2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      decayRadiusmin2.push_back(-10.);
    }
    decayRadiusMin = decayRadiusmin2;
  }
  // if decayRadiusMax size smaller than particleID , fill up further with defaults
  if (particleID.size() > decayRadiusMax.size()) {
    vector<double> decayRadiusmax2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      decayRadiusmax2.push_back(1.e5);
    }
    decayRadiusMax = decayRadiusmax2;
  }

  // if decayZMin size smaller than particleID , fill up further with defaults
  if (particleID.size() > decayZMin.size()) {
    vector<double> decayZmin2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      decayZmin2.push_back(-1.e5);
    }
    decayZMin = decayZmin2;
  }
  // if decayZMax size smaller than particleID , fill up further with defaults
  if (particleID.size() > decayZMax.size()) {
    vector<double> decayZmax2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      decayZmax2.push_back(1.e5);
    }
    decayZMax = decayZmax2;
  }

  // check if beta is smaller than 1
  if (std::abs(betaBoost) >= 1) {
    edm::LogError("MCSmartSingleParticleFilter") << "Input beta boost is >= 1 !";
  }
}

bool MCSmartSingleParticleFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    for (unsigned int i = 0; i < particleID.size(); i++) {
      if (particleID[i] == (*p)->pdg_id() || particleID[i] == 0) {
        if ((*p)->momentum().perp() > ptMin[i] && ((*p)->status() == status[i] || status[i] == 0)) {
          HepMC::FourVector mom = MCFilterZboostHelper::zboost((*p)->momentum(), betaBoost);
          if (mom.rho() > pMin[i] && mom.eta() > etaMin[i] && mom.eta() < etaMax[i]) {
            if (!((*p)->production_vertex()))
              continue;

            double decx = (*p)->production_vertex()->position().x();
            double decy = (*p)->production_vertex()->position().y();
            double decrad = sqrt(decx * decx + decy * decy);
            if (decrad < decayRadiusMin[i])
              continue;
            if (decrad > decayRadiusMax[i])
              continue;

            double decz = (*p)->production_vertex()->position().z();
            if (decz < decayZMin[i])
              continue;
            if (decz > decayZMax[i])
              continue;

            accepted = true;
          }
        }
      }
    }
  }
  return accepted;
}

DEFINE_FWK_MODULE(MCSmartSingleParticleFilter);
