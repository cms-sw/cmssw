// -*- C++ -*-
//
// Package:    DJpsiFilter
// Class:      DJpsiFilter
//
/**\class DJpsiFilter DJpsiFilter.cc psi2s1s/DJpsiFilter/src/DJpsiFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  bian jianguo
//         Created:  Tue Nov 22 20:39:54 CST 2011
//
//

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <cstdlib>
#include <string>

class DJpsiFilter : public edm::global::EDFilter<> {
public:
  explicit DJpsiFilter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const double minPt;
  const double maxY;
  const double maxPt;
  const double minY;
  const int status;
  const int particleID;
};

DJpsiFilter::DJpsiFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      minPt(iConfig.getUntrackedParameter("MinPt", 0.)),
      maxY(iConfig.getUntrackedParameter("MaxY", 10.)),
      maxPt(iConfig.getUntrackedParameter("MaxPt", 1000.)),
      minY(iConfig.getUntrackedParameter("MinY", 0.)),
      status(iConfig.getUntrackedParameter("Status", 0)),
      particleID(iConfig.getUntrackedParameter("ParticleID", 0)) {}

bool DJpsiFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  int n2jpsi = 0;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    if ((*p)->status() != status)
      continue;
    double energy = (*p)->momentum().e();
    double pz = (*p)->momentum().pz();
    double momentumY = 0.5 * std::log((energy + pz) / (energy - pz));
    if ((*p)->momentum().perp() > minPt && std::fabs(momentumY) < maxY && (*p)->momentum().perp() < maxPt &&
        std::fabs(momentumY) > minY) {
      if (std::abs((*p)->pdg_id()) == particleID)
        n2jpsi++;
    }
    if (n2jpsi >= 2) {
      accepted = true;
      break;
    }
  }
  return accepted;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DJpsiFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(DJpsiFilter);
