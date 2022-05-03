
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "GeneratorInterface/GenFilters/plugins/MCLongLivedParticles.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>
#include <vector>

using namespace edm;
using namespace std;

//Filter particles based on their minimum and/or maximum displacement on the transverse plane and optionally on their pdgIds
//To run independently of pdgId, do not insert the particleIDs entry in filter declaration

MCLongLivedParticles::MCLongLivedParticles(const edm::ParameterSet& iConfig):
      moduleLabel_(iConfig.getParameter<std::string>("hepMCProductTag")),
      hepMCProductTag_(edm::InputTag(iConfig.getUntrackedParameter(moduleLabel_, std::string("generator")), "unsmeared")),
      token_(consumes<edm::HepMCProduct>(hepMCProductTag_)),
      particleIDs(iConfig.getParameter<std::vector<int>>("ParticleIDs")),
      theUpperCut_(iConfig.getParameter<double>("LengMax")),
      theLowerCut_(iConfig.getParameter<double>("LengMin")){}

void MCLongLivedParticles::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
   edm::ParameterSetDescription desc;
   desc.add<std::string>("hepMCProductTag", "moduleLabel");
   desc.add<std::vector<int>>("ParticleIDs", std::vector<int>{0});
   desc.add<double>("LengMax", -1.);
   desc.add<double>("LengMin", -1.);
   descriptions.add("selectsDisplacement", desc);
}

// ------------ method called to skim the data  ------------
bool MCLongLivedParticles::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  
  Handle<HepMCProduct> evt;

  iEvent.getByToken(token_, evt);

  bool pass = false;
  bool matchedID = true;

  float theUpperCut2 = theUpperCut_ * theUpperCut_;
  float theLowerCut2 = theLowerCut_ * theLowerCut_;


  if (particleIDs.at(0) != 0) 
     matchedID = false;

  const HepMC::GenEvent* generated_event = evt->GetEvent();
  HepMC::GenEvent::particle_const_iterator p;

  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++) {
    //if a list of pdgId is provided, loop only on particles with those pdgId
     
  for (unsigned int idx = 0; idx < particleIDs.size(); idx++) {
        if (abs((*p)->pdg_id()) == abs(particleIDs.at(idx))) {  //compares absolute values of pdgIds
          matchedID = true;
          break;
        }
      }
    

    if (matchedID) {
       if (theLowerCut_ <= 0. && theUpperCut_ <= 0.)  {
          pass = true;
          break;
       }
      if (((*p)->production_vertex() != nullptr) && ((*p)->end_vertex() != nullptr)) {
        float dist2 = (((*p)->production_vertex())->position().x() - ((*p)->end_vertex())->position().x()) *
                          (((*p)->production_vertex())->position().x() - ((*p)->end_vertex())->position().x()) +
                      (((*p)->production_vertex())->position().y() - ((*p)->end_vertex())->position().y()) *
                          (((*p)->production_vertex())->position().y() - ((*p)->end_vertex())->position().y());

        if ((dist2 >= theLowerCut2 || theLowerCut_ <= 0.) &&
            (dist2 < theUpperCut2 ||
             theUpperCut_ <= 0.)) {  //lower cut can be also 0 - prompt particle needs to be accepted in that case
          pass = true;
          break;
        }
      }
      if (((*p)->production_vertex() == nullptr) && (!((*p)->end_vertex() != nullptr))) {
        if ((((*p)->end_vertex()->position().perp() >= theLowerCut_) || theLowerCut_ <= 0.) &&
            (((*p)->end_vertex()->position().perp() < theUpperCut_) ||
             theUpperCut_ <= 0.)) {  // lower cut can be also 0 - prompt particle needs to be accepted in that case
          pass = true;
          break;
        }
      }
    }
  } 

  return pass;
}
DEFINE_FWK_MODULE(MCLongLivedParticles);
