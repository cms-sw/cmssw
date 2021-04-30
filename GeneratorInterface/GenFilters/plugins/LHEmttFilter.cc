// -*- C++ -*-
//
// Package:    LHEmttFilter
// Class:      LHEmttFilter
//
/* 

 Description: Filter to select ttbar events with invariant mass over a certain threshold.
  (Based on LHEPtFilter)
 */

// system include files
#include <memory>
#include <iostream>
#include <set>

// user include files
#include "Math/Vector4D.h"
#include "Math/Vector4Dfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

//
// class declaration
//
class LHEmttFilter : public edm::global::EDFilter<> {
public:
  explicit LHEmttFilter(const edm::ParameterSet&);
  ~LHEmttFilter() override;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------

  edm::EDGetTokenT<LHEEventProduct> src_;
  double ptMin_ = 0;
  double MinInvMass_ = -1;
  double MaxInvMass_ = -1;
};

using namespace edm;
using namespace std;

LHEmttFilter::LHEmttFilter(const edm::ParameterSet& iConfig)
    : ptMin_(iConfig.getParameter<double>("ptMin")),
      MinInvMass_(iConfig.getParameter<double>("MinInvMass")),
      MaxInvMass_(iConfig.getParameter<double>("MaxInvMass")) {
  src_ = consumes<LHEEventProduct>(iConfig.getParameter<edm::InputTag>("src"));
}

LHEmttFilter::~LHEmttFilter() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to skim the data  ------------
bool LHEmttFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<LHEEventProduct> EvtHandle;
  iEvent.getByToken(src_, EvtHandle);

  std::vector<lhef::HEPEUP::FiveVector> lheParticles = EvtHandle->hepeup().PUP;
  std::vector<ROOT::Math::PxPyPzMVector> cands;
  std::vector<int> pdgId_cands;

  for (unsigned int i = 0; i < lheParticles.size(); i++) {
    if (EvtHandle->hepeup().ISTUP[i] != 2) {  // keep only intermediate particles
      continue;
    }
    int pdgId = EvtHandle->hepeup().IDUP[i];
    if (std::abs(pdgId) != 6) {  // keep only top quarks
      continue;
    }
    pdgId_cands.push_back(pdgId);
    cands.push_back(
        ROOT::Math::PxPyPzMVector(lheParticles[i][0], lheParticles[i][1], lheParticles[i][2], lheParticles[i][4]));
  }

  if (cands.size() != 2) {
    edm::LogWarning("LHEmttFilter Error") << "Number of top quarks found in the event != 2" << endl;
  }

  double ttmass_ = -1;
  if (!cands.empty() && (cands.size() == 2)) {                           //two top quarks
    if (pdgId_cands.at(0) + pdgId_cands.at(1) == 0) {                    //exactly one t and one tbar
      if ((cands.at(0).pt() > ptMin_) && (cands.at(1).pt() > ptMin_)) {  //requiring minimum pT
        ROOT::Math::PxPyPzMVector tot = cands.at(0);
        for (unsigned icand = 1; icand < cands.size(); ++icand) {
          tot += cands.at(icand);
        }
        ttmass_ = tot.mass();
      }
    } else if (pdgId_cands.at(0) + pdgId_cands.at(1) != 0) {
      edm::LogWarning("LHEmttFilter Error") << "Found two t/tbar quarks instead of a ttbar pair" << endl;
    }
  }

  if ((MinInvMass_ > -1 && ttmass_ < MinInvMass_) || (MaxInvMass_ > -1 && ttmass_ > MaxInvMass_)) {
    return false;
  } else {
    return true;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHEmttFilter);
