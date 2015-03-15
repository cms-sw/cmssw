#include "GeneratorInterface/GenFilters/interface/PythiaFilterEMJet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>
#include<list>
#include<map>
#include<vector>
#include<cmath>

using namespace edm;
using namespace std;

namespace{

  inline double deltaR2(double eta0, double phi0, double eta, double phi){
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


PythiaFilterEMJet::PythiaFilterEMJet(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")))),
etaMin(iConfig.getUntrackedParameter<double>("MinEMEta", 0)),
eTSumMin(iConfig.getUntrackedParameter<double>("ETSumMin", 50.)),
pTMin(iConfig.getUntrackedParameter<double>("MinEMpT", 5.)),
etaMax(iConfig.getUntrackedParameter<double>("MaxEMEta", 2.7)),
eTSumMax(iConfig.getUntrackedParameter<double>("ETSumMax", 100.)),
pTMax(iConfig.getUntrackedParameter<double>("MaxEMpT", 999999.)),
ebEtaMax(1.479),
maxnumberofeventsinrun(iConfig.getUntrackedParameter<int>("MaxEvents",3000000)){ 
  
  deltaEB=0.01745/2  *5; // delta_eta, delta_phi
  deltaEE=2.93/317/2 *5; // delta_x/z, delta_y/z
  theNumberOfTestedEvt = 0;
  theNumberOfSelected = 0;

  cout << " Max Events : " << maxnumberofeventsinrun << endl;
  cout << " Cut Definition: " << endl;
  cout << " MinEMEta = " << etaMin << endl;
  cout << " ETSumMin = " << eTSumMin << endl;
  cout << " MinEMpT = " << pTMin << endl;
  cout << " MaxEMEta = " << etaMax << endl;
  cout << " ETSumMax = " << eTSumMax << endl;
  cout << " MaxEMpT = " << pTMax << endl;
  cout << " 5x5 crystal cone  around EM axis in ECAL Barrel = " << deltaEB << endl;
  cout << " 5x5 crystal cone  around EM axis in ECAL Endcap = " << deltaEE << endl;

}

PythiaFilterEMJet::~PythiaFilterEMJet()
{
std::cout << "Total number of tested events = " << theNumberOfTestedEvt << std::endl;
std::cout << "Total number of accepted events = " << theNumberOfSelected << std::endl;
}


// ------------ method called to produce the data  ------------
bool PythiaFilterEMJet::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){

  if(theNumberOfSelected>=maxnumberofeventsinrun)   {
    throw cms::Exception("endJob") << "we have reached the maximum number of events ";
  }
  
  theNumberOfTestedEvt++;
  if(theNumberOfTestedEvt%1000 == 0) cout << "Number of tested events = " << theNumberOfTestedEvt <<  endl;
  
  bool accepted = false;
  Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  list<const HepMC::GenParticle *> EM_seeds;
  const HepMC::GenEvent * myGenEvent = evt->GetEvent();

  int particle_id = 1;

  //select e+/e-/gamma particles in the events
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();  
                                                 p != myGenEvent->particles_end(); 
						 ++p ) {
   
			  
    if ( (abs((*p)->pdg_id())==11 || (*p)->pdg_id()==22)  
	 && (*p)->status()==1
	 && (*p)->momentum().perp() > pTMin
	 && (*p)->momentum().perp() < pTMax 
	 && fabs((*p)->momentum().eta()) < etaMax  
	 && fabs((*p)->momentum().eta()) > etaMin ) {
      EM_seeds.push_back(*p);
    }
    particle_id++;
  }

  EM_seeds.sort(ParticlePtGreater());
  
 double etaEMClus=0;
 double phiEMClus=0;
 double ptEMClus=0;
 for(std::list<const HepMC::GenParticle *>::const_iterator is=EM_seeds.begin(); 
                                                           is!= EM_seeds.end(); 
							   ++is){
    double etaEM=(*is)->momentum().eta();
    double phiEM=(*is)->momentum().phi();
    double ptEM=(*is)->momentum().perp();
    //pass at the cluster the seed infos
    etaEMClus = etaEM;
    phiEMClus = phiEM;
    ptEMClus = ptEM;

    //check if the EM particle is in the barrel
    bool inEB(0);
    double tgx(0);
    double tgy(0);
    if( std::abs(etaEM)<ebEtaMax ) inEB=1;
    else{
      tgx=(*is)->momentum().px()/(*is)->momentum().pz();
      tgy=(*is)->momentum().py()/(*is)->momentum().pz();
    }
    
//   std::vector<const HepMC::GenParticle*> takenEM ; 
//   std::vector<const HepMC::GenParticle*>::const_iterator itPart ;

    for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   
                                                   p != myGenEvent->particles_end(); 
						   ++p ) {
    
      if (((*p)->status()==1 && (*p)->pdg_id() == 22) ||       // gamma
	  ((*p)->status()==1 && abs((*p)->pdg_id()) == 11) )   // electron
// 	   (*p)->pdg_id() == 111 ||                  	       // pi0
// 	   abs((*p)->pdg_id()) == 221 ||             	       // eta
// 	   abs((*p)->pdg_id()) == 331 ||             	       // eta prime
// 	   abs((*p)->pdg_id()) == 113 ||             	       // rho0 
// 	   abs((*p)->pdg_id()) == 223  )             	       // omega*/
        {
// 	 // check if found is daughter of one already taken
// 	 bool isUsed = false ;
// 	 const HepMC::GenParticle* mother = (*p)->production_vertex() ?
//                                      *((*p)->production_vertex()->particles_in_const_begin()) : 0 ;
// 	 const HepMC::GenParticle* motherMother = (mother != 0  && mother->production_vertex()) ?
//                                      *(mother->production_vertex()->particles_in_const_begin()) : 0 ;
// 	 const HepMC::GenParticle* motherMotherMother = (motherMother != 0 && motherMother->production_vertex()) ?
//                                      *(motherMother->production_vertex()->particles_in_const_begin()) : 0 ;
// 	 for(itPart = takenEM.begin(); itPart != takenEM.end(); ++itPart) {
// 	 if ((*itPart) == mother ||
// 	     (*itPart) == motherMother ||
// 	     (*itPart) == motherMotherMother) 
// 	     {
// 	       isUsed = true ;
// 	       break ;	    
// 	     }
// 	  }
// 	 if (!isUsed) takenEM.push_back(*p);

	 double pt=(*p)->momentum().perp();
	 if (pt == ptEM) continue ;              //discard the same particle of the seed
	 double eta=(*p)->momentum().eta();
	 double phi=(*p)->momentum().phi();

	 if(inEB) {
	   if(  std::abs(eta-etaEM)> deltaEB ||
		std::abs(deltaPhi(phi,phiEM)) > deltaEB) continue;
	  }
	 else if( std::abs((*p)->momentum().px()/(*p)->momentum().pz() - tgx)
		  > deltaEE || 
		  std::abs((*p)->momentum().py()/(*p)->momentum().pz() - tgy)
		  > deltaEE) continue;
	 ptEMClus  += pt ;
	 if(inEB)
	   {
	     etaEMClus += (eta-etaEMClus)*pt/ptEMClus ;
	     phiEMClus += deltaPhi(phi,phiEM)*pt/ptEMClus;
	   }
	 else
           {
	     etaEMClus += ((*p)->momentum().px()/(*p)->momentum().pz() - tgx)*pt/ptEMClus ;
	     phiEMClus += ((*p)->momentum().py()/(*p)->momentum().pz() - tgy)*pt/ptEMClus;
	   }
	 }
      }
     if( ptEMClus > eTSumMin) 
         accepted = true ;       
   }

  if (accepted) {
    theNumberOfSelected++;
    cout << "========>  Event preselected " << theNumberOfSelected
         << " Proccess ID " << myGenEvent->signal_process_id() << endl;
    return true; 
  }
  else return false;
}

