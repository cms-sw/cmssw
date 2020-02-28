// -*- C++ -*-
//
// Package:    ZgammaMassFilter
// Class:      ZgammaMassFilter
//
/*

 Description: filter events based on the Pythia particle information

 Implementation: inherits from generic EDFilter

*/
//
// Original Author:  Alexey Ferapontov
//         Created:  Thu July 26 11:57:54 CDT 2012
// $Id: ZgammaMassFilter.h,v 1.1 2012/08/10 12:46:29 lenzip Exp $
//
//

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TLorentzVector.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

class ZgammaMassFilter : public edm::global::EDFilter<> {
public:
  explicit ZgammaMassFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;

  const double minPhotonPt;
  const double minLeptonPt;

  const double minPhotonEta;
  const double minLeptonEta;

  const double maxPhotonEta;
  const double maxLeptonEta;

  const double minDileptonMass;
  const double minZgMass;
};

namespace {
  // order std::vector of TLorentzVector elements
  class orderByPt {
  public:
    bool operator()(TLorentzVector const& a, TLorentzVector const& b) {
      if (a.Pt() == b.Pt()) {
        return a.Pt() < b.Pt();
      } else {
        return a.Pt() > b.Pt();
      }
    }
  };
}  // namespace

using namespace edm;
using namespace std;

ZgammaMassFilter::ZgammaMassFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(iConfig.getParameter<InputTag>("HepMCProduct"))),
      minPhotonPt(iConfig.getParameter<double>("minPhotonPt")),
      minLeptonPt(iConfig.getParameter<double>("minLeptonPt")),
      minPhotonEta(iConfig.getParameter<double>("minPhotonEta")),
      minLeptonEta(iConfig.getParameter<double>("minLeptonEta")),
      maxPhotonEta(iConfig.getParameter<double>("maxPhotonEta")),
      maxLeptonEta(iConfig.getParameter<double>("maxLeptonEta")),
      minDileptonMass(iConfig.getParameter<double>("minDileptonMass")),
      minZgMass(iConfig.getParameter<double>("minZgMass")) {}

bool ZgammaMassFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  vector<TLorentzVector> Lepton;
  Lepton.clear();
  vector<TLorentzVector> Photon;
  Photon.clear();
  vector<float> Charge;
  Charge.clear();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if ((*p)->status() == 1 && (abs((*p)->pdg_id()) == 11 || abs((*p)->pdg_id()) == 13 || abs((*p)->pdg_id()) == 15)) {
      TLorentzVector LeptP((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e());
      if (LeptP.Pt() > minLeptonPt) {
        Lepton.push_back(LeptP);
      }  // if pt
    }    // if lepton

    if (abs((*p)->pdg_id()) == 22 && (*p)->status() == 1) {
      TLorentzVector PhotP((*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e());
      if (PhotP.Pt() > minPhotonPt) {
        Photon.push_back(PhotP);
      }  // if pt
    }    // if photon

  }  // loop over particles

  // std::cout << "\n" << "Photon size: " << Photon.size() << std::endl;
  // for (unsigned int u=0; u<Photon.size(); u++){
  //   std::cout << "BEF photon PT: " << Photon[u].Pt() << std::endl;
  // }
  // std::cout << "\n" << "Lepton size: " << Lepton.size() << std::endl;
  // for (unsigned int u=0; u<Lepton.size(); u++){
  //   std::cout << "BEF lepton PT: " << Lepton[u].Pt() << std::endl;
  // }

  // order Lepton and Photon according to Pt
  std::stable_sort(Photon.begin(), Photon.end(), orderByPt());
  std::stable_sort(Lepton.begin(), Lepton.end(), orderByPt());

  //  std::cout << "\n" << std::endl;
  //  std::cout << "\n" << "Photon size: " << Photon.size() << std::endl;
  //  for (unsigned int u=0; u<Photon.size(); u++){
  //    std::cout << "AFT photon PT: " << Photon[u].Pt() << std::endl;
  //  }
  //  std::cout << "\n" << "Lepton size: " << Lepton.size() << std::endl;
  //  for (unsigned int u=0; u<Lepton.size(); u++){
  //    std::cout << "AFT lepton PT: " << Lepton[u].Pt() << std::endl;
  //  }
  //  std::cout << "\n" << std::endl;

  if (!Photon.empty() && Lepton.size() > 1 && Photon[0].Pt() > minPhotonPt && Lepton[0].Pt() > minLeptonPt &&
      Lepton[1].Pt() > minLeptonPt && Photon[0].Eta() > minPhotonEta && Lepton[0].Eta() > minLeptonEta &&
      Lepton[1].Eta() > minLeptonEta && Photon[0].Eta() < maxPhotonEta && Lepton[0].Eta() < maxLeptonEta &&
      Lepton[1].Eta() < maxLeptonEta && (Lepton[0] + Lepton[1]).M() > minDileptonMass &&
      (Lepton[0] + Lepton[1] + Photon[0]).M() >
          minZgMass) {  // satisfy molteplicity, kinematics, and ll llg minimum mass
    accepted = true;
  }

  //  std::cout << "++ returning: " << accepted << "\n" << std::endl;

  return accepted;
}

DEFINE_FWK_MODULE(ZgammaMassFilter);
