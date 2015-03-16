#include "GeneratorInterface/GenFilters/interface/BdecayFilter.h"

using namespace edm;
using namespace std;
using namespace HepMC;



BdecayFilter::BdecayFilter(const edm::ParameterSet& iConfig)
{
  token_ = consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")));
  motherParticle = iConfig.getParameter< int >("motherParticle");

  firstDaughter.type = iConfig.getParameter< int >("firstDaughter");
  firstDaughter.decayProduct   = iConfig.getParameter< vector<int> >("firstDaughterDecay");
  firstDaughter.etaMin = iConfig.getParameter<double>("firstDaughterDecayEtaMin");
  firstDaughter.etaMax = iConfig.getParameter<double>("firstDaughterDecayEtaMax");
  firstDaughter.ptMin = iConfig.getParameter<double>("firstDaughterDecayPtMin");

  secondDaughter.type = iConfig.getParameter< int >("secondDaughter");
  secondDaughter.decayProduct   = iConfig.getParameter< vector<int> >("secondDaughterDecay");
  secondDaughter.etaMin = iConfig.getParameter<double>("secondDaughterDecayEtaMin");
  secondDaughter.etaMax = iConfig.getParameter<double>("secondDaughterDecayEtaMax");
  secondDaughter.ptMin = iConfig.getParameter<double>("secondDaughterDecayPtMin");

  noAccepted = 0;
}


BdecayFilter::~BdecayFilter()
{  
  std::cout << "Total number of accepted events = " << noAccepted << std::endl;
}

/*
HepMC::GenParticle * BdecayFilter::findParticle(const GenPartVect genPartVect,
	const int requested_id)
{
  for (GenPartVectIt p = genPartVect.begin(); p != genPartVect.end(); p++)
    {
      int event_particle_id = abs( (*p)->pdg_id() );
  cout << "isC "<<event_particle_id<<"\n";
      if (requested_id == event_particle_id) return *p;
    }
  return 0;
}
*/

HepMC::GenParticle * BdecayFilter::findParticle(HepMC::GenVertex* vertex, 
						   const int requested_id)
{
  // for(std::set<GenParticle*>::const_iterator p = vertex->particles_out_const_begin(); 
  for(GenVertex::particles_out_const_iterator  p = vertex->particles_out_const_begin(); 
      p != vertex->particles_out_const_end(); p++)
    {
      int event_particle_id = abs( (*p)->pdg_id() );
      cout << "particle Id: "<<event_particle_id<<"\n";
      if (requested_id == event_particle_id) return *p;
    }
  return 0;
}

HepMC::GenEvent::particle_const_iterator BdecayFilter::getNextBs(const HepMC::GenEvent::particle_const_iterator start, const HepMC::GenEvent::particle_const_iterator end)
{
  HepMC::GenEvent::particle_const_iterator p;
  for (p = start; p != end; p++) 
    {
      int event_particle_id = abs( (*p)->pdg_id() );
//   cout << "search "<<event_particle_id<<"\n";
      if (event_particle_id == motherParticle) return p;
    }
  return p;  
}


bool BdecayFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  
  const HepMC::GenEvent * generated_event = evt->GetEvent();
  //cout << "Start\n";

  bool event_passed = false;
  HepMC::GenEvent::particle_const_iterator bs = getNextBs(generated_event->particles_begin(), generated_event->particles_end());
  while (bs!=  generated_event->particles_end() ) {

    // vector< GenParticle * > bsChild = (*bs)->listChildren();

    //***
    HepMC::GenVertex* outVertex = (*bs)->end_vertex();
    //***
    
    GenParticle * jpsi = 0;
    GenParticle * phi = 0;
    // cout << "bs size "<<bsChild.size()<<endl;
    //***
    int numChildren = outVertex->particles_out_size();
    cout<< "B children "<<numChildren<<endl;
    //***
    
    /*    if ((bsChild.size()==2) && ((jpsi = findParticle(bsChild, 443))!=0) && 
	  ((phi = findParticle(bsChild, 333))!=0)) {
	  cout << bsChild[0]->momentum()<<" "<<bsChild[0]->momentum().eta()
	  <<" "<<bsChild[1]->momentum()<<" "<<bsChild[1]->momentum().eta()<<endl;
    */
    
    //***
    if( (numChildren==2) && ((jpsi = findParticle(outVertex, firstDaughter.type))!=0) && 
	((phi = findParticle(outVertex, secondDaughter.type))!=0)) {
      
      cout << jpsi->momentum().rho() <<" "<<jpsi->momentum().eta() <<" "<<phi->momentum().rho()<<" "<<phi->momentum().eta()<<endl;
      //cout <<"bs dec trouve"<<endl;
      if (cuts(phi, secondDaughter) && cuts(jpsi, firstDaughter)) {
        cout <<"decay found"<<endl;
        event_passed = true;
        break;
      }
    }
    bs = getNextBs(++bs, generated_event->particles_end());
  }
  
  if (event_passed) noAccepted++;
  cout << "End filter\n";
  
  delete generated_event; 
  
  return event_passed;
}


bool BdecayFilter::cuts(const GenParticle * jpsi, const CutStruct& cut)
{
	cout << "start cuts" << endl;
  HepMC::GenVertex* myVertex = jpsi->end_vertex();
  int numChildren = myVertex->particles_out_size();
  int numDecProd = cut.decayProduct.size();
  std::vector<HepMC::GenParticle*> psiChild;
  // for(std::set<GenParticle*>::const_iterator p = myVertex->particles_out_const_begin();
  for(GenVertex::particles_out_const_iterator p = myVertex->particles_out_const_begin();
      p != myVertex->particles_out_const_end(); p++) 
    psiChild.push_back((*p));

  if (numChildren!=numDecProd) return false;
    cout << psiChild[0]->pdg_id()<<" "<<psiChild[1]->pdg_id()<<endl;

  for (int i=0; i<numChildren; ++i) {
    bool goodPart = false;
    for (int j=0; j < numChildren; ++j){
      if (abs(psiChild[i]->pdg_id()) == abs(cut.decayProduct[j])) goodPart = true;
	}
	cout << psiChild[i]->momentum().perp() << endl;
    if ( !goodPart || (!etaInRange(psiChild[i]->momentum().eta(), cut.etaMin, cut.etaMax)) || (psiChild[i]->momentum().perp() < cut.ptMin) ) return false;
  }
	cout << "cuts true" << endl;
  return true;
}

bool BdecayFilter::etaInRange(float eta, float etamin, float etamax)
{
  return ( (etamin < eta) && (eta < etamax) );
}
