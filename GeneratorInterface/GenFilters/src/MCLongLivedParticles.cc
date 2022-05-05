
#include "GeneratorInterface/GenFilters/interface/MCLongLivedParticles.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <vector>

using namespace edm;
using namespace std;

//Filter particles based on their minimum and/or maximum displacement on the transverse plane and optionally on their pdgIds
//To run independently of pdgId, do not insert the particleIDs entry in filter declaration

MCLongLivedParticles::MCLongLivedParticles(const edm::ParameterSet& iConfig) :
  //hepMCProductTag is left untracked for backwards compatibility
  hepMCProductTag_(iConfig.getUntrackedParameter<edm::InputTag>("hepMCProductTag",edm::InputTag("generator","unsmeared"))),
  token_(consumes<edm::HepMCProduct>(hepMCProductTag_)),
  particleIDs_(iConfig.getParameter<std::vector<int>>("ParticleIDs")),
  //theCut is left untracked for backwards compatibility
  theCut(iConfig.getUntrackedParameter<double>("LengCut",10.)), // for backwards compatibility
  theUpperCut_(iConfig.getParameter<double>("LengMax")),
  theLowerCut_(iConfig.getParameter<double>("LengMin"))
{}


void MCLongLivedParticles::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<int>>("ParticleIDs", std::vector<int>{0});
  desc.add<double>("LengMax", -1.);
  desc.add<double>("LengMin", -1.);
}


// ------------ method called to skim the data  ------------
bool MCLongLivedParticles::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  
  Handle<HepMCProduct> evt;
  
  iEvent.getByToken(token_, evt);
    
  bool pass = false;
  bool matchedID = true;

  //backwards compatibility with previous single lower cut filter: if theCut is well defined and theUpperCut and theLowerCut are undefined
  // revert to previous behavior (lower cut only)
  if (theCut >=0 && (theUpperCut_< 0 && theLowerCut_< 0) ){    
 	theUpperCut_ = -1.;
	theLowerCut_ = theCut;
  }

  
  const float theUpperCut2 = theUpperCut_*theUpperCut_;
  const float theLowerCut2 = theLowerCut_*theLowerCut_;
  
  const HepMC::GenEvent * generated_event = evt->GetEvent();
  HepMC::GenEvent::particle_const_iterator p;
  
  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++)
    { 
      //if a list of pdgId is provided, loop only on particles with those pdgId 
   if (particleIDs_.at(0)!=0) matchedID = false;
        
	for (unsigned int idx=0; idx < particleIDs_.size(); idx++){
          if (abs((*p)->pdg_id())==abs(particleIDs_.at(idx))){ //compares absolute values of pdgIds
            matchedID = true;
            break;
          }
        }                
      
      if (matchedID){

	if (theLowerCut_ <= 0. && theUpperCut_ <= 0. && theCut <= 0.)  {
	   pass = true;
	   break;
	} 

        if(((*p)->production_vertex() != nullptr) && ((*p)->end_vertex()!=nullptr)){
        
          float dist2 = (((*p)->production_vertex())->position().x()-((*p)->end_vertex())->position().x())*(((*p)->production_vertex())->position().x()-((*p)->end_vertex())->position().x()) +
                        (((*p)->production_vertex())->position().y()-((*p)->end_vertex())->position().y())*(((*p)->production_vertex())->position().y()-((*p)->end_vertex())->position().y());
          //lower cut can be also 0 - prompt particle needs to be accepted in that case
	  if( (dist2>=theLowerCut2 || theLowerCut_<=0.) && 
              (dist2< theUpperCut2 || theUpperCut_<=0.) ){ 
            pass=true;
            break;
	    }
        }
        if(((*p)->production_vertex() == nullptr) && (!((*p)->end_vertex() == nullptr))) { 
          // lower cut can be also 0 - prompt particle needs to be accepted in that case
          float distEndVert = (*p)->end_vertex()->position().perp();
          if(((distEndVert >= theLowerCut_) || theLowerCut_<=0.) && 
             ((distEndVert <  theUpperCut_) || theUpperCut_<=0.)){ 	 
	    pass=true;
            break;
	   }
        }
      }
    }
     
  return pass;
}

