// -*- C++ -*-
//
// Package:    PythiaHLTSoupFilter
// Class:      PythiaHLTSoupFilter
//
/**\class PythiaHLTSoupFilter PythiaHLTSoupFilter.cc IOMC/PythiaHLTSoupFilter/src/PythiaHLTSoupFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Jan 23 14:57:54 CET 2006
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

#include <cmath>
#include <cstdlib>
#include <string>

class PythiaHLTSoupFilter : public edm::global::EDFilter<> {
public:
  explicit PythiaHLTSoupFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;

  const double minptelectron;
  const double minptmuon;
  const double maxetaelectron;
  const double maxetamuon;
  const double minpttau;
  const double maxetatau;
};

using namespace std;

PythiaHLTSoupFilter::PythiaHLTSoupFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      minptelectron(iConfig.getUntrackedParameter("MinPtElectron", 0.)),
      minptmuon(iConfig.getUntrackedParameter("MinPtMuon", 0.)),
      maxetaelectron(iConfig.getUntrackedParameter("MaxEtaElectron", 10.)),
      maxetamuon(iConfig.getUntrackedParameter("MaxEtaMuon", 10.)),
      minpttau(iConfig.getUntrackedParameter("MinPtTau", 0.)),
      maxetatau(iConfig.getUntrackedParameter("MaxEtaTau", 10.)) {}

bool PythiaHLTSoupFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  if (myGenEvent->signal_process_id() == 2) {
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p) {
      if (abs((*p)->pdg_id()) == 11 && (*p)->momentum().perp() > minptelectron &&
          abs((*p)->momentum().eta()) < maxetaelectron && (*p)->status() == 1) {
        accepted = true;
      }

      if (abs((*p)->pdg_id()) == 13 && (*p)->momentum().perp() > minptmuon &&
          abs((*p)->momentum().eta()) < maxetamuon && (*p)->status() == 1) {
        accepted = true;
      }

      if (abs((*p)->pdg_id()) == 15 && (*p)->momentum().perp() > minpttau && abs((*p)->momentum().eta()) < maxetatau &&
          (*p)->status() == 3) {
        accepted = true;
      }
    }

  } else {
    accepted = true;
  }
  return accepted;
}

DEFINE_FWK_MODULE(PythiaHLTSoupFilter);
