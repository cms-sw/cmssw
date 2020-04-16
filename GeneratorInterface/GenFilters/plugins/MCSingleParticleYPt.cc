// -*- C++ -*-
//
// Package:    MCSingleParticleYPt
// Class:      MCSingleParticleYPt
//
/*

 Description: filter events based on the Pythia particleID, Pt, Y and status. It is based on MCSingleParticleFilter.
              It will used to filter a b-hadron with the given kinematics, only one b-hadron is required to match.
 Implementation: inherits from generic EDFilter

*/
//
// Author: Alberto Sanchez-Hernandez
// Adapted on: August 2016
//
//
// Filter based on MCSingleParticleFilter.cc, but using rapidity instead of eta

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <ostream>
#include <string>
#include <vector>

class MCSingleParticleYPt : public edm::global::EDFilter<> {
public:
  explicit MCSingleParticleYPt(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const int fVerbose;
  const bool fchekantiparticle;
  std::vector<int> particleID;
  std::vector<double> ptMin;
  std::vector<double> rapMin;
  std::vector<double> rapMax;
  std::vector<int> status;
};

using namespace edm;
using namespace std;

MCSingleParticleYPt::MCSingleParticleYPt(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      fVerbose(iConfig.getUntrackedParameter("verbose", 0)),
      fchekantiparticle(iConfig.getUntrackedParameter("CheckAntiparticle", true)) {
  vector<int> defpid;
  defpid.push_back(0);
  particleID = iConfig.getUntrackedParameter<vector<int> >("ParticleID", defpid);
  vector<double> defptmin;
  defptmin.push_back(0.);
  ptMin = iConfig.getUntrackedParameter<vector<double> >("MinPt", defptmin);
  vector<double> defrapmin;
  defrapmin.push_back(-10.);
  rapMin = iConfig.getUntrackedParameter<vector<double> >("MinY", defrapmin);
  vector<double> defrapmax;
  defrapmax.push_back(10.);
  rapMax = iConfig.getUntrackedParameter<vector<double> >("MaxY", defrapmax);
  vector<int> defstat;
  defstat.push_back(0);
  status = iConfig.getUntrackedParameter<vector<int> >("Status", defstat);

  // check for same size
  if ((ptMin.size() > 1 && particleID.size() != ptMin.size()) ||
      (rapMin.size() > 1 && particleID.size() != rapMin.size()) ||
      (rapMax.size() > 1 && particleID.size() != rapMax.size()) ||
      (status.size() > 1 && particleID.size() != status.size())) {
    edm::LogWarning("MCSingleParticleYPt")
        << "WARNING: MCSingleParticleYPt : size of vector cuts do not match!!" << endl;
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
  if (particleID.size() > rapMin.size()) {
    vector<double> defrapmin2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      defrapmin2.push_back(-10.);
    }
    rapMin = defrapmin2;
  }
  // if etaMax size smaller than particleID , fill up further with defaults
  if (particleID.size() > rapMax.size()) {
    vector<double> defrapmax2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      defrapmax2.push_back(10.);
    }
    rapMax = defrapmax2;
  }
  // if status size smaller than particleID , fill up further with defaults
  if (particleID.size() > status.size()) {
    vector<int> defstat2;
    for (unsigned int i = 0; i < particleID.size(); i++) {
      defstat2.push_back(0);
    }
    status = defstat2;
  }

  if (fVerbose > 0) {
    edm::LogInfo("MCSingleParticleYPt") << "----------------------------------------------------------------------"
                                        << std::endl;
    edm::LogInfo("MCSingleParticleYPt") << "----- MCSingleParticleYPt" << std::endl;
    for (unsigned int i = 0; i < particleID.size(); ++i) {
      edm::LogInfo("MCSingleParticleYPt") << " ID: " << particleID[i] << " pT > " << ptMin[i] << ",   " << rapMin[i]
                                          << " < y < " << rapMax[i] << ",   status = " << status[i] << std::endl;
    }
    if (fchekantiparticle)
      edm::LogInfo("MCSingleParticleYPt") << " anti-particles will be tested as well." << std::endl;
    edm::LogInfo("MCSingleParticleYPt") << "----------------------------------------------------------------------"
                                        << std::endl;
  }
}

bool MCSingleParticleYPt::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();
  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if (fVerbose > 3)
      edm::LogInfo("MCSingleParticleYPt")
          << "Looking at particle : " << (*p)->pdg_id() << " status : " << (*p)->status() << std::endl;

    for (unsigned int i = 0; i < particleID.size(); i++) {
      if (particleID[i] == (*p)->pdg_id() || (fchekantiparticle && (-particleID[i] == (*p)->pdg_id())) ||
          particleID[i] == 0) {
        // calculate rapidity just for the desired particle and make sure, this particles has enough energy
        double rapidity = ((*p)->momentum().e() - (*p)->momentum().pz()) > 0.
                              ? 0.5 * log(((*p)->momentum().e() + (*p)->momentum().pz()) /
                                          ((*p)->momentum().e() - (*p)->momentum().pz()))
                              : rapMax[i] + .1;
        if (fVerbose > 2)
          edm::LogInfo("MCSingleParticleYPt")
              << "Testing particle : " << (*p)->pdg_id() << " pT: " << (*p)->momentum().perp() << " y: " << rapidity
              << " status : " << (*p)->status() << endl;
        if ((*p)->momentum().perp() > ptMin[i] && rapidity > rapMin[i] && rapidity < rapMax[i] &&
            ((*p)->status() == status[i] || status[i] == 0)) {
          accepted = true;
          if (fVerbose > 1)
            edm::LogInfo("MCSingleParticleYPt")
                << "Accepted particle : " << (*p)->pdg_id() << " pT: " << (*p)->momentum().perp() << " y: " << rapidity
                << " status : " << (*p)->status() << endl;
          break;
        }
      }
    }
    if (accepted)
      break;
  }
  return accepted;
}

DEFINE_FWK_MODULE(MCSingleParticleYPt);
