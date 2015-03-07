#include "GeneratorInterface/GenFilters/interface/PythiaFilterEMJetHeep.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//#include "CLHEP/HepMC/GenParticle.h"

#include <iostream>
#include<list>
#include<vector>
#include<cmath>
#include "TFile.h"

// using namespace edm;
// using namespace std;
// using namespace HepMC;


//namespace{

double PythiaFilterEMJetHeep::deltaR(double eta0, double phi0, double eta, double phi){
    double dphi=phi-phi0;
    if(dphi>M_PI) dphi-=2*M_PI;
    else if(dphi<=-M_PI) dphi+=2*M_PI;
    return sqrt(dphi*dphi+(eta-eta0)*(eta-eta0));
}

struct ParticlePtGreater{
      bool operator()(const HepMC::GenParticle * a,
                      const HepMC::GenParticle * b) const{
      return a->momentum().perp() > b->momentum().perp();
    }
};

//}

PythiaFilterEMJetHeep::PythiaFilterEMJetHeep(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("source")))),
//
minEventPt(iConfig.getUntrackedParameter<double>("MinEventPt",40.)),
etaMax(iConfig.getUntrackedParameter<double>("MaxEta", 2.8)),
cone_clust(iConfig.getUntrackedParameter<double>("ConeClust",0.10)), 
cone_iso(iConfig.getUntrackedParameter<double>("ConeIso",0.50)), 
nPartMin(iConfig.getUntrackedParameter<unsigned int>("NumPartMin", 2)),
drMin(iConfig.getUntrackedParameter<double>("dRMin", 0.4)),
//
eventsProcessed(0),
maxnumberofeventsinrun(iConfig.getUntrackedParameter<int>("MaxEvents",1000)),
outputFile_(iConfig.getUntrackedParameter("outputFile",std::string("rootOutputFile"))),
minbias(iConfig.getUntrackedParameter<bool>("Minbias",false)),
debug(iConfig.getUntrackedParameter<bool>("Debug",true)) { 
theNumberOfSelected = 0;   
//  
}

PythiaFilterEMJetHeep::~PythiaFilterEMJetHeep(){}

void PythiaFilterEMJetHeep::beginJob() {

  // parametarizations of presel. criteria:

  ptSeedMin_EB =  6.0 + (minEventPt - 80.) * 0.035;
  ptSeedMin_EE =  5.5 + (minEventPt - 80.) * 0.033;
  fracConePtMin_EB = 0.60 - (minEventPt - 80.) * 0.0009;
  fracConePtMin_EE = fracConePtMin_EB;
  fracEmPtMin_EB = 0.30 + (minEventPt - 80.) * 0.0017;
  if(minEventPt >=225) fracEmPtMin_EB = 0.40 + (minEventPt - 230.) * 0.00063;
  fracEmPtMin_EE = 0.30 + (minEventPt - 80.) * 0.0033;
  if(minEventPt >=165) fracEmPtMin_EE = 0.55 + (minEventPt - 170.) * 0.0005;
  fracTrkPtMax_EB = 0.80 - (minEventPt - 80.) * 0.001;
  fracTrkPtMax_EE = 0.70 - (minEventPt - 80.) * 0.001;
  isoConeMax_EB = 0.35;
  isoConeMax_EE = 0.40;
  ptHdMax_EB = 40;
  ptHdMax_EB = 45;
  ntrkMax_EB = 4;
  ntrkMax_EE = 4;

  if( minbias ) {

    std::cout <<" ... Minbias preselection ... " << std::endl;

    minEventPt = 1.0;
    ptSeedMin_EB =  1.5; 
    ptSeedMin_EE =  1.5; 
    fracConePtMin_EB = 0.20; 
    fracConePtMin_EE = fracConePtMin_EB;
    fracEmPtMin_EB = 0.20; 
    fracEmPtMin_EE = 0.20; 
    fracTrkPtMax_EB = 0.80; 
    fracTrkPtMax_EE = 0.80; 
    isoConeMax_EB = 1.0;
    isoConeMax_EE = 1.0;
    ptHdMax_EB = 7;
    ptHdMax_EB = 7;
    ntrkMax_EB = 2;
    ntrkMax_EE = 2;

  }


  std::cout <<" minEventPt = " << minEventPt << std::endl;
  std::cout <<" etaMax = " << etaMax << std::endl;
  std::cout <<" nPartMin = " << nPartMin << std::endl;
  std::cout <<" cone_clust = " << cone_clust << std::endl;
  std::cout <<" cone_iso = " << cone_iso << std::endl;
  std::cout <<" drMin = " << drMin << std::endl << std::endl;
  std::cout <<" ptSeedMin_EB = " << ptSeedMin_EB << std::endl;
  std::cout <<" ptSeedMin_EE = " << ptSeedMin_EE << std::endl;
  std::cout <<" fracConePtMin_EB = " << fracConePtMin_EB << std::endl;
  std::cout <<" fracConePtMin_EE = " << fracConePtMin_EE << std::endl;
  std::cout <<" fracEmPtMin_EB = " << fracEmPtMin_EB << std::endl;
  std::cout <<" fracEmPtMin_EE = " << fracEmPtMin_EE << std::endl;
  std::cout <<" fracTrkPtMax_EB = " << fracTrkPtMax_EB << std::endl;
  std::cout <<" fracTrkPtMax_EE = " << fracTrkPtMax_EE << std::endl;
  std::cout <<" ntrkMax_EB = " << ntrkMax_EB << std::endl;
  std::cout <<" ntrkMax_EE = " << ntrkMax_EE << std::endl;
  std::cout <<" ptHdMax_EB = " << ptHdMax_EB << std::endl;
  std::cout <<" ptHdMax_EE = " << ptHdMax_EE << std::endl;
  std::cout <<" isoConeMax_EB = " << isoConeMax_EB << std::endl;
  std::cout <<" isoConeMax_EE = " << isoConeMax_EE << std::endl;

}

// ------------ method called to produce the data  ------------
bool PythiaFilterEMJetHeep::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){

  if(theNumberOfSelected>=maxnumberofeventsinrun)   {
    throw cms::Exception("endJob")<<"we have reached the maximum number of events ";
  }

  accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  const HepMC::GenEvent * myGenEvent = evt->GetEvent();

  eventsProcessed++;

  std::list<const HepMC::GenParticle *> seeds;
  std::list<const HepMC::GenParticle *> candidates;
 
// collect the seeds
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p ) {
   
    double pt=(*p)->momentum().perp();
    double eta=(*p)->momentum().eta();

    int pdgid = (*p)->pdg_id(); 
    if ( !(pdgid==22 || abs(pdgid)==11 || abs(pdgid)==211 || abs(pdgid)==321) ) continue;
    //if ( !(pdgid==22 || abs(pdgid)==11) ) continue;
 
//  Selection #1: seed particle ETA
    if ( std::abs(eta) > etaMax ) continue;
    bool EB = false;
    bool EE = false;
    if (fabs(eta)<1.5)                         EB = true;
    else if (fabs(eta)>=1.5&&fabs(eta)<etaMax) EE = true;   
    if (debug) std::cout << " Selection 1 passed " << std::endl;

//  Selection #2: seed particle pT
    if ( EB && pt < ptSeedMin_EB ) continue;
    if ( EE && pt < ptSeedMin_EE ) continue;
    if (debug) std::cout << " Selection 2 passed " << std::endl;

    seeds.push_back(*p);   
  }
  
  if (seeds.size() < nPartMin ) return false;

  seeds.sort(ParticlePtGreater());

// select the proto-clusters
  std::list<const HepMC::GenParticle*>::iterator itSeed;

  for(itSeed = seeds.begin(); itSeed != seeds.end(); ++itSeed) {

    double pt=(*itSeed)->momentum().perp();
    double eta=(*itSeed)->momentum().eta();
    double phi=(*itSeed)->momentum().phi();  

    bool EB = false;
    bool EE = false;
    if (fabs(eta)<1.5)                         EB = true;
    else if (fabs(eta)>=1.5&&fabs(eta)<etaMax) EE = true;     

    float setCone_iso=0;
    float setCone_clust=0;
    float setEM=0;
    float setHAD=0;
    float ptMaxHadron=0;
    float setCharged=0;
    unsigned int Ncharged = 0;
    
    for ( HepMC::GenEvent::particle_const_iterator pp = myGenEvent->particles_begin();  pp != myGenEvent->particles_end(); ++pp ) {
    
      if ( (*pp)->status()!=1 ) continue;  // select just stable particles
      int pid= (*pp)->pdg_id();
      int apid= std::abs(pid);
      if(apid==310) apid = 22; 
      if (apid > 11 && apid < 20) continue; //get rid of muons and neutrinos

      double pt_ =(*pp)->momentum().perp();
      double eta_=(*pp)->momentum().eta();
      double phi_=(*pp)->momentum().phi();

      float dr = deltaR(eta_, phi_, eta, phi);
      if ( dr <= cone_iso )  setCone_iso += pt_;

      bool charged = false;
      if ( apid ==211 || apid == 321 || apid == 2212 || apid == 11 ) charged = true;
      if ( dr <= cone_clust ) {
         setCone_clust += pt_;
         if ( apid == 22 || apid == 11) setEM += pt_;
         if ( apid>100 )   setHAD += pt_;
	 if ( apid>100 && pt_ > ptMaxHadron ) ptMaxHadron = pt_;
         if ( charged && pt_ > 1 ) { Ncharged++; setCharged += pt_;}
      }     
    }
    setCone_iso -= setCone_clust;

    if ( pt / setCone_clust < 0.15 ) continue;

    // Selection #3: min. pT of all particles in the proto-cluster   
    if (EB && setCone_clust < fracConePtMin_EB*minEventPt ) continue;
    if (EE && setCone_clust < fracConePtMin_EE*minEventPt ) continue;
    if (debug) std::cout << " Selection 3 passed " << std::endl;

     // Selections #4: min/max pT fractions of e/gamma in the proto-cluster from the total pT 
    if ( EB && setEM / setCone_clust  < fracEmPtMin_EB ) continue;
    if ( EE && setEM / setCone_clust  < fracEmPtMin_EE ) continue;
    if (debug) std::cout << " Selection 4 passed " << std::endl;

    // Selection 5: max. track pT fractions and number of tracks in the proto-cluster
    if ( (EB && setCharged / setCone_clust > fracTrkPtMax_EB) ||  Ncharged > ntrkMax_EB) continue;
    if ( (EE && setCharged / setCone_clust > fracTrkPtMax_EE) ||  Ncharged > ntrkMax_EE) continue;
    if (debug) std::cout << " Selection 5 passed " << std::endl;

    // Selection #6: max. pT of the hadron in the proto-cluster
    if ( EB && ptMaxHadron > ptHdMax_EB ) continue;
    if ( EE && ptMaxHadron > ptHdMax_EE ) continue;
    if (debug) std::cout << " Selection 6 passed " << std::endl;

    // Selection #7: max. pT fraction in the dR=cone_iso around the proto-cluster
    if ( EB && setCone_iso / setCone_clust > isoConeMax_EB) continue;
    if ( EE && setCone_iso / setCone_clust > isoConeMax_EE) continue;
    if (debug) std::cout << " Selection 7 passed " << std::endl;       

    if (debug) {
      std::cout <<  "(*itSeed)->pdg_id() = " << (*itSeed)->pdg_id() << std::endl;
      std::cout <<  "pt = " << pt << std::endl;
      std::cout <<  "setCone_clust = " << setCone_clust << std::endl;
      std::cout <<  "setEM = " << setEM << std::endl;
      std::cout <<  "ptMaxHadron = " << ptMaxHadron << std::endl;
      std::cout <<  "setCone_iso = " << setCone_iso << std::endl;
      std::cout <<  "setCone_iso / setCone_clust = " << setCone_iso / setCone_clust << std::endl;    
    }

    // Selection #8: any two objects should be separated by dr > drMin
    bool same = false;
    std::list<const HepMC::GenParticle*>::iterator itPart;
    for(itPart = candidates.begin(); itPart != candidates.end(); ++itPart) {
      float eta_ = (*itPart)->momentum().eta();
      float phi_ = (*itPart)->momentum().phi();
      float dr = deltaR(eta_, phi_, eta, phi);    
      if (dr < drMin ) same = true;
    }
    if (same) continue;
    if (debug) std::cout << " Selection 8 passed " << std::endl;       

    candidates.push_back(*itSeed);

  }  

  if (candidates.size() >= nPartMin ) {
    accepted=true;
    theNumberOfSelected++;
  }

  if (debug) std::cout<<" Event preselected "<<theNumberOfSelected<<" Proccess ID "<<myGenEvent->signal_process_id()<<std::endl;

  return accepted; 
  
}

void PythiaFilterEMJetHeep::endJob() {

  std::cout<<"\n *************************\n "<<std::endl;
  std::cout<<" Events processed "<< eventsProcessed <<std::endl;
  std::cout<<" Events preselected "<< theNumberOfSelected <<std::endl;
  std::cout<<"\n ************************* \n"<<std::endl;

}

