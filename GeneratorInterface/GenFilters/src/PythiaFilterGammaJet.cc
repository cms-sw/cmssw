#include "GeneratorInterface/GenFilters/interface/PythiaFilterGammaJet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
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


PythiaFilterGammaJet::PythiaFilterGammaJet(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")))),
etaMax(iConfig.getUntrackedParameter<double>("MaxPhotonEta", 2.8)),
ptSeed(iConfig.getUntrackedParameter<double>("PhotonSeedPt", 5.)),
ptMin(iConfig.getUntrackedParameter<double>("MinPhotonPt")),
ptMax(iConfig.getUntrackedParameter<double>("MaxPhotonPt")),
dphiMin(iConfig.getUntrackedParameter<double>("MinDeltaPhi", -1)/180*M_PI),
detaMax(iConfig.getUntrackedParameter<double>("MaxDeltaEta", 10.)),
etaPhotonCut2(iConfig.getUntrackedParameter<double>("MinPhotonEtaForwardJet", 1.3)),
cone(0.5),ebEtaMax(1.479),
maxnumberofeventsinrun(iConfig.getUntrackedParameter<int>("MaxEvents",10000)){ 
  
  deltaEB=0.01745/2  *5; // delta_eta, delta_phi
  deltaEE=2.93/317/2 *5; // delta_x/z, delta_y/z
  theNumberOfSelected = 0;
}


PythiaFilterGammaJet::~PythiaFilterGammaJet(){}


// ------------ method called to produce the data  ------------
bool PythiaFilterGammaJet::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){

  if(theNumberOfSelected>=maxnumberofeventsinrun)   {
    throw cms::Exception("endJob")<<"we have reached the maximum number of events ";
  }

  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  std::list<const HepMC::GenParticle *> seeds;
  const HepMC::GenEvent * myGenEvent = evt->GetEvent();

  if(myGenEvent->signal_process_id() == 14 || myGenEvent->signal_process_id() == 29) {


  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p ) {
   
    if ( (*p)->pdg_id()==22 && (*p)->status()==1
	 && (*p)->momentum().perp() > ptSeed 
	 && std::abs((*p)->momentum().eta()) < etaMax  ) seeds.push_back(*p);

  }

  seeds.sort(ParticlePtGreater());
  for(std::list<const HepMC::GenParticle *>::const_iterator is=
	seeds.begin(); is!=seeds.end(); is++){
 
    double etaPhoton=(*is)->momentum().eta();
    double phiPhoton=(*is)->momentum().phi();

    HepMC::GenEvent::particle_const_iterator ppp = myGenEvent->particles_begin();
    for(int i=0;i<6;++i) ppp++;
    HepMC::GenParticle* particle7 = (*ppp);
    ppp++;
    HepMC::GenParticle* particle8 = (*ppp);

    double dphi7=std::abs(deltaPhi(phiPhoton, 
				   particle7->momentum().phi()));
    double dphi=std::abs(deltaPhi(phiPhoton, 
				  particle8->momentum().phi()));
    int jetline=8;
    if(dphi7>dphi) {
      dphi=dphi7;
      jetline=7;
    }
    if(dphi<dphiMin) continue;
    //double etaJet= myGenEvent->particle(jetline)->momentum().eta();
    double etaJet = 0.0;
    if(jetline==8) etaJet = particle8->momentum().eta();
    else etaJet = particle7->momentum().eta();

    double eta1=etaJet-detaMax;
    double eta2=etaJet+detaMax;
    if (eta1>etaPhotonCut2) eta1=etaPhotonCut2;
    if (eta2<-etaPhotonCut2) eta2=-etaPhotonCut2;
    if (etaPhoton<eta1 ||etaPhoton>eta2) continue;

    bool inEB(0);
    double tgx(0);
    double tgy(0);
    if( std::abs(etaPhoton)<ebEtaMax) inEB=1;
    else{
      tgx=(*is)->momentum().px()/(*is)->momentum().pz();
      tgy=(*is)->momentum().py()/(*is)->momentum().pz();
    }

    double etPhoton=0;
    double etPhotonCharged=0;
    double etCone=0;
    double etConeCharged=0;
    double ptMaxHadron=0;
    
    
    for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p ) {
    
      if ( (*p)->status()!=1 ) continue; 
      int pid= (*p)->pdg_id();
      int apid= std::abs(pid);
      if (apid>11 &&  apid<20) continue; //get rid of muons and neutrinos
      double eta=(*p)->momentum().eta();
      double phi=(*p)->momentum().phi();
      if (deltaR2(etaPhoton, phiPhoton, eta, phi)>cone*cone) continue;
      double pt=(*p)->momentum().perp();


      edm::ESHandle<ParticleDataTable> pdt;
      iSetup.getData( pdt );


//       double charge=(*p)->particledata().charge();
      //int charge3=(*p)->particleID().threeCharge();

      int charge3 = ((pdt->particle((*p)->pdg_id()))->ID().threeCharge());
      etCone+=pt;
      if(charge3 && pt<2) etConeCharged+=pt;

      //select particles matching a crystal array centered on photon
      if(inEB) {
	if(  std::abs(eta-etaPhoton)> deltaEB ||
	     std::abs(deltaPhi(phi,phiPhoton)) > deltaEB) continue;
      }
      else if( std::abs((*p)->momentum().px()/(*p)->momentum().pz() - tgx)
	       > deltaEE || 
	       std::abs((*p)->momentum().py()/(*p)->momentum().pz() - tgy)
	       > deltaEE) continue;

      etPhoton+=pt;
      if(charge3 && pt<2) etPhotonCharged+=pt;
      if(apid>100 && apid!=310 && pt>ptMaxHadron) ptMaxHadron=pt;

    }

    if(etPhoton<ptMin ||etPhoton>ptMax) continue;

    //isolation cuts
    if(etCone-etPhoton> 5+etPhoton/20-etPhoton*etPhoton/1e4) continue;
    if(etCone-etPhoton-(etConeCharged-etPhotonCharged) > 
       3+etPhoton/20-etPhoton*etPhoton*etPhoton/1e6) continue;
    if(ptMaxHadron > 4.5+etPhoton/40) continue;
    
    accepted=true;
    break;

  } //loop over seeds
  
  } else {
  // end of if(gammajetevent)
  return true;
  // accept all non-gammajet events
  }
  
  if (accepted) {
    theNumberOfSelected++;
    return true; 
  }
  else return false;

}

