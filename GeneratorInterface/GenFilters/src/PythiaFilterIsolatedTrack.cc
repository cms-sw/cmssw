#include "GeneratorInterface/GenFilters/interface/PythiaFilterIsolatedTrack.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <iostream>
#include<list>
#include<vector>
#include<cmath>

//using namespace edm;
//using namespace std;

namespace{

  double deltaR2(double eta0, double phi0, double eta, double phi){
    double dphi=phi-phi0;
    if(dphi>M_PI) dphi-=2*M_PI;
    else if(dphi<=-M_PI) dphi+=2*M_PI;
    return dphi*dphi+(eta-eta0)*(eta-eta0);
  }

  double deltaPhi(double phi0, double phi){
    double dphi=phi-phi0;
    if(dphi>M_PI) dphi-=2*M_PI;
    else if(dphi<=-M_PI) dphi+=2*M_PI;
    return dphi;
  }

  class ParticlePtGreater{
  public:
    int operator()(const HepMC::GenParticle * p1, 
		   const HepMC::GenParticle * p2) const{
      return p1->momentum().perp() > p2->momentum().perp();
    }
  };
}


PythiaFilterIsolatedTrack::PythiaFilterIsolatedTrack(const edm::ParameterSet& iConfig) :
label_(iConfig.getUntrackedParameter("moduleLabel",std::string("source"))),
etaMax(iConfig.getUntrackedParameter<double>("MaxChargedHadronEta", 2.3)),
ptSeed(iConfig.getUntrackedParameter<double>("ChargedHadronSeedPt", 10.)),
cone(iConfig.getUntrackedParameter<double>("isoCone", 0.5)),
ebEtaMax(1.479)
{ 
  
  deltaEB=0.01745/2  *5; // delta_eta, delta_phi
  deltaEE=2.93/317/2 *5; // delta_x/z, delta_y/z
  ebEtaMax=1.479;
  theNumberOfSelected = 0;
}


PythiaFilterIsolatedTrack::~PythiaFilterIsolatedTrack(){}


// ------------ method called to produce the data  ------------
bool PythiaFilterIsolatedTrack::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){

//  if(theNumberOfSelected>=maxnumberofeventsinrun)   {
//    throw cms::Exception("endJob")<<"we have reached the maximum number of events ";
//  }
//  std::cout<<" Start PythiaFilterIsolatedTrack::filter "<<std::endl;
  
  edm::ESHandle<ParticleDataTable> pdt;
  iSetup.getData( pdt );

  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByLabel(label_, evt);

  std::list<const HepMC::GenParticle *> seeds;
  const HepMC::GenEvent * myGenEvent = evt->GetEvent();

   std::vector<HepMC::GenEvent::particle_const_iterator> seed_itr;
   
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p ) {
  
    if (abs((*p)->pdg_id())>11 && abs((*p)->pdg_id()) <21) continue;
    
    int charge3 = ((pdt->particle((*p)->pdg_id()))->ID().threeCharge());
    if ( abs(charge3) == 3 && (*p)->status()==1
	 && (*p)->momentum().perp() > ptSeed 
	 && std::abs((*p)->momentum().eta()) < etaMax  ) { 
	        // std::cout<<" Pid "<<(*p)->pdg_id()<<" "<<(*p)->momentum().eta()<<" "<<(*p)->momentum().phi()
	        //                                          <<" "<<(*p)->momentum().perp()<<std::endl;
							   seeds.push_back(*p);seed_itr.push_back(p);
							 }
  }

  seeds.sort(ParticlePtGreater());
  
//  for(std::list<const HepMC::GenParticle *>::const_iterator is=
//	seeds.begin(); is!=seeds.end(); is++){
  for( std::vector<HepMC::GenEvent::particle_const_iterator>::iterator is=seed_itr.begin(); 
                                                             is != seed_itr.end(); 
							     is++){
     
     
 
    double etaChargedHadron=(**is)->momentum().eta();
    double phiChargedHadron=(**is)->momentum().phi();

    bool inEB(0);
    double tgx(0);
    double tgy(0);
    if( std::abs(etaChargedHadron)<ebEtaMax) inEB=1;
    else{
      tgx=(**is)->momentum().px()/(**is)->momentum().pz();
      tgy=(**is)->momentum().py()/(**is)->momentum().pz();
    }

    double etChargedHadron=(**is)->momentum().perp();
    double etChargedHadronCharged=0;
    double etCone=0;
    double etConeCharged=0;
    double ptMaxHadron=0;

//    std::cout<<" Start seed "<< (**is)->pdg_id()<<" pt "<<(**is)->momentum().perp()<<std::endl;  
//    if(abs((**is)->pdg_id()) == 11)
//    {
//        HepMC::GenVertex::particles_in_const_iterator inbegin =  (**is)->production_vertex()->particles_in_const_begin();
//        HepMC::GenVertex::particles_in_const_iterator inend = (**is)->production_vertex()->particles_in_const_end();
//	for(HepMC::GenVertex::particles_in_const_iterator ii = inbegin; ii != inend; ii++)
//	{
//	  std::cout<<" Parent for electron "<<(*ii)->pdg_id()<<std::endl;
//	}
//    }  
    
    for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p ) {
    
    
      if ( (*p)->status()!=1 ) continue; 
      int pid= (*p)->pdg_id();
      int apid= std::abs(pid);
      if (apid>11 &&  apid<21) continue; //get rid of muons and neutrinos
      double eta=(*p)->momentum().eta();
      double phi=(*p)->momentum().phi();
      if (deltaR2(etaChargedHadron, phiChargedHadron, eta, phi)>cone*cone) continue;
      
      double pt=(*p)->momentum().perp();
      
      //***
      
            
      int charge3 = ((pdt->particle((*p)->pdg_id()))->ID().threeCharge());
      //***

      etCone+=pt;

      if(charge3 && pt<2) {etConeCharged+=pt;}

       // Calculate the max Pt hadron in the cone (except pre-selected)
       
      if( (*is) == p)
      {
// Do not take seed for calculation ptMaxHadron      
//         std::cout<<"Found id "<<(*p)->pdg_id()<<" "<<(*p)->momentum().perp() <<std::endl;
	 continue;
      }
      
      if(apid>100 && apid!=310 && pt>ptMaxHadron) {
          ptMaxHadron=pt; 
//	  std::cout<<" PtMax "<<pid<<" "<<pt<<" "<<deltaR2(etaChargedHadron, phiChargedHadron, eta, phi)<<std::endl;
	  }

    }

    //isolation cuts

    double isocut1 = 5+etChargedHadron/20-etChargedHadron*etChargedHadron/1e4;
    double isocut2 = 3+etChargedHadron/20-etChargedHadron*etChargedHadron*etChargedHadron/1e6;
    double isocut3 = 4.5+etChargedHadron/40;
    if (etChargedHadron>165.)
    {
      isocut1 = 5.+165./20.-165.*165./1e4;
      isocut2 = 3.+165./20.-165.*165.*165./1e6;
      isocut3 = 4.5+165./40.;
    }

//    std::cout<<" etCone "<<etCone<<" "<<etChargedHadron<<" "<<etCone-etChargedHadron<<" "<<isocut1<<std::endl;
//    std::cout<<"Second cut on iso "<<etCone-etChargedHadron-(etConeCharged-etChargedHadronCharged)<<" cut value "<<isocut2<<" etChargedHadron "<<etChargedHadron<<std::endl;
//    std::cout<<" PtHadron "<<ptMaxHadron<<" "<<4.5+etChargedHadron/40<<std::endl;
    
    if(etCone-etChargedHadron > isocut1) continue;
    if(ptMaxHadron > isocut3) continue;
    
//    std::cout<<"Accept event "<<abs((**is)->pdg_id())<<" pt "<<(**is)->momentum().perp()<<std::endl;
    accepted=true;
    break;

  } //loop over seeds
  
  
  if (accepted) {
    theNumberOfSelected++;
    std::cout<<" Event preselected "<<theNumberOfSelected<<" Proccess ID "<<myGenEvent->signal_process_id()<<std::endl;
    return true; 
  }
  else return false;

}

