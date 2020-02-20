// -*- C++ -*-
//
// Package:    MCZll
// Class:      MCZll
//
/*

 Description: filter events based on the Pythia ProcessID and the Pt_hat

 Implementation: inherits from generic EDFilter

*/
//
// Original Author:  Paolo Meridiani
//
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
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

class MCZll : public edm::global::EDFilter<> {
public:
  explicit MCZll(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  int leptonFlavour_;
  double leptonPtMin_;
  double leptonPtMax_;
  double leptonEtaMin_;
  double leptonEtaMax_;
  std::pair<double, double> zMassRange_;
  bool filter_;
};

using namespace edm;
using namespace std;

MCZll::MCZll(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))) {
  leptonFlavour_ = iConfig.getUntrackedParameter<int>("leptonFlavour", 11);
  leptonPtMin_ = iConfig.getUntrackedParameter<double>("leptonPtMin", 5.);
  leptonPtMax_ = iConfig.getUntrackedParameter<double>("leptonPtMax", 99999.);
  leptonEtaMin_ = iConfig.getUntrackedParameter<double>("leptonEtaMin", 0.);
  leptonEtaMax_ = iConfig.getUntrackedParameter<double>("leptonEtaMax", 2.7);
  zMassRange_.first = iConfig.getUntrackedParameter<double>("zMassMin", 60.);
  zMassRange_.second = iConfig.getUntrackedParameter<double>("zMassMax", 120.);
  filter_ = iConfig.getUntrackedParameter<bool>("filter", true);
  std::ostringstream str;
  str << "=========================================================\n"
      << "Filter MCZll being constructed with parameters: "
      << "\nleptonFlavour " << leptonFlavour_ << "\nleptonPtMin " << leptonPtMin_ << "\nleptonPtMax " << leptonPtMax_
      << "\nleptonEtaMin " << leptonEtaMin_ << "\nleptonEtaMax " << leptonEtaMax_ << "\nzMassMin " << zMassRange_.first
      << "\nzMassMax " << zMassRange_.second << "\n=========================================================";
  edm::LogVerbatim("MCZllInfo") << str.str();
  if (filter_)
    produces<HepMCProduct>();
}

bool MCZll::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  std::unique_ptr<HepMCProduct> bare_product(new HepMCProduct());

  bool accepted = false;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  HepMC::GenEvent* myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));
  HepMC::GenEvent* zEvent = new HepMC::GenEvent();

  if (myGenEvent->signal_process_id() != 1) {
    delete myGenEvent;
    delete zEvent;
    return false;
  }

  //found a prompt Z

  for (HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p) {
    if (!accepted && ((*p)->pdg_id() == 23) && (*p)->status() == 3) {
      accepted = true;
      HepMC::GenVertex* zVertex = new HepMC::GenVertex();
      HepMC::GenParticle* myZ = new HepMC::GenParticle(*(*p));
      zVertex->add_particle_in(myZ);
      //	  std::cout << (*p)->momentum().invariantMass() << std::endl;
      if ((*p)->momentum().m() < zMassRange_.first || (*p)->momentum().m() > zMassRange_.second)
        accepted = false;
      std::vector<HepMC::GenParticle*> children;
      HepMC::GenVertex* outVertex = (*p)->end_vertex();
      for (HepMC::GenVertex::particles_out_const_iterator iter = outVertex->particles_out_const_begin();
           iter != outVertex->particles_out_const_end();
           iter++)
        children.push_back(*iter);
      std::vector<HepMC::GenParticle*>::const_iterator aDaughter;
      for (aDaughter = children.begin(); aDaughter != children.end(); aDaughter++) {
        HepMC::GenParticle* myDa = new HepMC::GenParticle(*(*aDaughter));
        zVertex->add_particle_out(myDa);
        if ((*aDaughter)->status() == 2)
          continue;
        //	      (*aDaughter)->print();

        if (!(abs((*aDaughter)->pdg_id()) == abs(leptonFlavour_)))
          accepted = false;
        //		std::cout << (*aDaughter)->momentum().perp() << " " << (*aDaughter)->momentum().eta() << std::endl;
        if ((*aDaughter)->momentum().perp() < leptonPtMin_)
          accepted = false;
        if ((*aDaughter)->momentum().perp() > leptonPtMax_)
          accepted = false;
        if (fabs((*aDaughter)->momentum().eta()) > leptonEtaMax_)
          accepted = false;
        if (fabs((*aDaughter)->momentum().eta()) < leptonEtaMin_)
          accepted = false;
      }
      zEvent->add_vertex(zVertex);
      if (accepted)
        break;
    }
  }

  if (accepted) {
    if (zEvent)
      bare_product->addHepMCData(zEvent);
    if (filter_)
      iEvent.put(std::move(bare_product));
    LogDebug("MCZll") << "Event " << iEvent.id().event() << " accepted" << std::endl;
    delete myGenEvent;
    return true;
  }

  delete myGenEvent;
  delete zEvent;
  return false;
}

DEFINE_FWK_MODULE(MCZll);
