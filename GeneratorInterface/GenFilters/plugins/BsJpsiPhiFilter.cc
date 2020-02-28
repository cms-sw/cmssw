#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HepMC/GenVertex.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

class BsJpsiPhiFilter : public edm::global::EDFilter<> {
public:
  explicit BsJpsiPhiFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  struct CutStruct {
    int type;
    double etaMin, etaMax, ptMin;
  };

  HepMC::GenParticle* findParticle(HepMC::GenVertex*, const int requested_id) const;

  HepMC::GenEvent::particle_const_iterator getNextBs(const HepMC::GenEvent::particle_const_iterator start,
                                                     const HepMC::GenEvent::particle_const_iterator end) const;

  bool cuts(const HepMC::GenParticle* jpsi, const CutStruct& cut) const;
  bool etaInRange(float eta, float etamin, float etamax) const;

  CutStruct leptonCuts, hadronCuts;

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
};

using namespace std;

BsJpsiPhiFilter::BsJpsiPhiFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))) {
  hadronCuts.type = iConfig.getParameter<int>("hadronType");
  hadronCuts.etaMin = iConfig.getParameter<double>("hadronEtaMin");
  hadronCuts.etaMax = iConfig.getParameter<double>("hadronEtaMax");
  hadronCuts.ptMin = iConfig.getParameter<double>("hadronPtMin");
  leptonCuts.type = iConfig.getParameter<int>("leptonType");
  leptonCuts.etaMin = iConfig.getParameter<double>("leptonEtaMin");
  leptonCuts.etaMax = iConfig.getParameter<double>("leptonEtaMax");
  leptonCuts.ptMin = iConfig.getParameter<double>("leptonPtMin");
}

HepMC::GenParticle* BsJpsiPhiFilter::findParticle(HepMC::GenVertex* vertex, const int requested_id) const {
  for (std::vector<HepMC::GenParticle*>::const_iterator p = vertex->particles_out_const_begin();
       p != vertex->particles_out_const_end();
       p++) {
    int event_particle_id = abs((*p)->pdg_id());
    if (requested_id == event_particle_id)
      return *p;
  }
  return nullptr;
}

HepMC::GenEvent::particle_const_iterator BsJpsiPhiFilter::getNextBs(
    const HepMC::GenEvent::particle_const_iterator start, const HepMC::GenEvent::particle_const_iterator end) const {
  HepMC::GenEvent::particle_const_iterator p;
  for (p = start; p != end; p++) {
    int event_particle_id = abs((*p)->pdg_id());
    if (event_particle_id == 531)
      return p;
  }
  return p;
}

bool BsJpsiPhiFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent* generated_event = evt->GetEvent();

  bool event_passed = false;
  HepMC::GenEvent::particle_const_iterator bs =
      getNextBs(generated_event->particles_begin(), generated_event->particles_end());
  while (bs != generated_event->particles_end()) {
    HepMC::GenVertex* outVertex = (*bs)->end_vertex();

    HepMC::GenParticle* jpsi = nullptr;
    HepMC::GenParticle* phi = nullptr;
    int numChildren = outVertex->particles_out_size();

    if ((numChildren == 2) && ((jpsi = findParticle(outVertex, 443)) != nullptr) &&
        ((phi = findParticle(outVertex, 333)) != nullptr)) {
      if (cuts(phi, hadronCuts) && cuts(jpsi, leptonCuts)) {
        event_passed = true;
        break;
      }
    }
    bs = getNextBs(++bs, generated_event->particles_end());
  }

  delete generated_event;

  return event_passed;
}

bool BsJpsiPhiFilter::cuts(const HepMC::GenParticle* jpsi, const CutStruct& cut) const {
  HepMC::GenVertex* myVertex = jpsi->end_vertex();
  int numChildren = myVertex->particles_out_size();
  std::vector<HepMC::GenParticle*> psiChild;
  for (std::vector<HepMC::GenParticle*>::const_iterator p = myVertex->particles_out_const_begin();
       p != myVertex->particles_out_const_end();
       p++)
    psiChild.push_back((*p));

  if (numChildren > 1) {
    if (psiChild.size() == 2 && (abs(psiChild[0]->pdg_id()) == cut.type) && (abs(psiChild[1]->pdg_id()) == cut.type)) {
      return ((etaInRange(psiChild[0]->momentum().eta(), cut.etaMin, cut.etaMax)) &&
              (etaInRange(psiChild[1]->momentum().eta(), cut.etaMin, cut.etaMax)) &&
              (psiChild[0]->momentum().perp() > cut.ptMin) && (psiChild[1]->momentum().perp() > cut.ptMin));
    }
    return false;
  }
  return false;
}

bool BsJpsiPhiFilter::etaInRange(float eta, float etamin, float etamax) const {
  return ((etamin < eta) && (eta < etamax));
}

DEFINE_FWK_MODULE(BsJpsiPhiFilter);
