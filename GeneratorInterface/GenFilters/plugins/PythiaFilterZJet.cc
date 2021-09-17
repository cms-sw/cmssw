/** \class PythiaFilterZJet
 *
 *  PythiaFilterZJet filter implements generator-level preselections
 *  for photon+jet like events to be used in jet energy calibration.
 *  Ported from fortran code written by V.Konoplianikov.
 *
 * \author A.Ulyanov, ITEP
 *
 ************************************************************/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
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
#include <vector>

class PythiaFilterZJet : public edm::global::EDFilter<> {
public:
  explicit PythiaFilterZJet(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const double etaMuMax;
  const double ptZMin;
  const double ptZMax;
};

PythiaFilterZJet::PythiaFilterZJet(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))),
      etaMuMax(iConfig.getUntrackedParameter<double>("MaxMuonEta", 2.5)),
      ptZMin(iConfig.getUntrackedParameter<double>("MinZPt")),
      ptZMax(iConfig.getUntrackedParameter<double>("MaxZPt")) {}

bool PythiaFilterZJet::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  if (myGenEvent->signal_process_id() == 15 || myGenEvent->signal_process_id() == 30) {
    std::vector<const HepMC::GenParticle*> mu;

    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p) {
      if (std::abs((*p)->pdg_id()) == 13 && (*p)->status() == 1)
        mu.push_back(*p);
      if (mu.size() > 1)
        break;
    }

    if (mu.size() > 1) {
      math::XYZTLorentzVector tot_mom(mu[0]->momentum());
      math::XYZTLorentzVector mom2(mu[1]->momentum());
      tot_mom += mom2;
      //    double ptZ= (mu[0]->momentum() + mu[1]->momentum()).perp();
      double ptZ = tot_mom.pt();
      if (ptZ > ptZMin && ptZ < ptZMax && std::abs(mu[0]->momentum().eta()) < etaMuMax &&
          std::abs(mu[1]->momentum().eta()) < etaMuMax)
        accepted = true;
    }

  } else {
    // end of if(gammajetevent)
    return true;
    // accept all non-gammajet events
  }
  return accepted;
}

DEFINE_FWK_MODULE(PythiaFilterZJet);
