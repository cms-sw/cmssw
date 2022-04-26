
#include "GeneratorInterface/GenFilters/interface/MCLongLivedParticles.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;

//Filter particles based on their minimum and/or maximum displacement on the transverse plane and optionally on their pdgIds
//To run independently of pdgId, do not insert the particleIDs entry in filter declaration

MCLongLivedParticles::MCLongLivedParticles(const edm::ParameterSet& iConfig) :
  token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"))),
  particleIDs(iConfig.getUntrackedParameter("ParticleIDs",std::vector<int>{0})),
  //hepMCProductTag_(iConfig.getUntrackedParameter<edm::InputTag>("hepMCProductTag",edm::InputTag("generator","unsmeared"))) {}
   //here do whatever other initialization is needed
  theCut(iConfig.getUntrackedParameter<double>("LengCut",-1.)),
  theUpperCut(iConfig.getUntrackedParameter<double>("LengMax",-1.)),
  theLowerCut(iConfig.getUntrackedParameter<double>("LengMin",-1.))
{}


MCLongLivedParticles::~MCLongLivedParticles()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to skim the data  ------------
bool MCLongLivedParticles::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  
  Handle<HepMCProduct> evt;
  
  iEvent.getByToken(token_, evt);
    
  bool pass = false;
  bool matchedID = true;

  if (theCut >=0){ //to restore previous behavior for single lower cut 
	theUpperCut = -1.;
	theLowerCut = theCut;
  }

  float theUpperCut2 = theUpperCut*theUpperCut;
  float theLowerCut2 = theLowerCut*theLowerCut;
  
  const HepMC::GenEvent * generated_event = evt->GetEvent();
  HepMC::GenEvent::particle_const_iterator p;
  
  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++)
    { 
      //if a list of pdgId is provided, loop only on particles with those pdgId 
      if (particleIDs.at(0)!=0){
        matchedID = false;
        for (unsigned int idx=0; idx < particleIDs.size(); idx++){
          if (abs((*p)->pdg_id())==abs(particleIDs.at(idx))){ //compares absolute values of pdgIds
            matchedID = true;
            break;
          }
        }                
      } 
      
      if (matchedID){ 

        if(((*p)->production_vertex() != nullptr) && ((*p)->end_vertex()!=nullptr)){
        
          float dist2 = (((*p)->production_vertex())->position().x()-((*p)->end_vertex())->position().x())*(((*p)->production_vertex())->position().x()-((*p)->end_vertex())->position().x()) +
                        (((*p)->production_vertex())->position().y()-((*p)->end_vertex())->position().y())*(((*p)->production_vertex())->position().y()-((*p)->end_vertex())->position().y());
       
      std::cout << (*p)->pdg_id() << " "  << sqrt(dist2)  << " " << theLowerCut  << " " << theUpperCut << std::endl;   
          if( (dist2>=theLowerCut2 || theLowerCut<=0.) && 
              (dist2< theUpperCut2 || theUpperCut<=0.) ){ //lower cut can be also 0 - prompt particle needs to be accepted in that case
      std::cout <<"__________________________" <<  (*p)->pdg_id() << " "  << sqrt(dist2)  << " " << theLowerCut  << " " << theUpperCut << std::endl;   
            pass=true;
            break;
	    }
        }
         
        if(((*p)->production_vertex()==nullptr) && (!((*p)->end_vertex()!=nullptr))){
      std::cout << (*p)->pdg_id() << " "  << (*p)->end_vertex()->position().perp()  << " " << theLowerCut  << " " << theUpperCut << std::endl;   
          if((((*p)->end_vertex()->position().perp() >= theLowerCut) || theLowerCut<=0.) && 
             (((*p)->end_vertex()->position().perp() <  theUpperCut) || theUpperCut<=0.)){ // lower cut can be also 0 - prompt particle needs to be accepted in that case
      	std::cout <<"____________________________" <<(*p)->pdg_id() << " "  << (*p)->end_vertex()->position().perp()  << " " << theLowerCut  << " " << theUpperCut << std::endl;   
            pass=true;
            break;
	   }
        }
      }
    }
  std::cout << "#################################################      "<< pass << std::endl; 
  return pass;
}

