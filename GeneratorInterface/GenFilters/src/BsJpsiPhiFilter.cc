#include "GeneratorInterface/GenFilters/interface/BsJpsiPhiFilter.h"

using namespace edm;
using namespace std;
using namespace HepMC;



BsJpsiPhiFilter::BsJpsiPhiFilter(const edm::ParameterSet& iConfig)
{
  token_ = consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"));
  hadronCuts.type = iConfig.getParameter< int >("hadronType");
  hadronCuts.etaMin = iConfig.getParameter<double>("hadronEtaMin");
  hadronCuts.etaMax = iConfig.getParameter<double>("hadronEtaMax");
  hadronCuts.ptMin = iConfig.getParameter<double>("hadronPtMin");
  leptonCuts.type = iConfig.getParameter< int >("leptonType");
  leptonCuts.etaMin = iConfig.getParameter<double>("leptonEtaMin");
  leptonCuts.etaMax = iConfig.getParameter<double>("leptonEtaMax");
  leptonCuts.ptMin = iConfig.getParameter<double>("leptonPtMin");

  noAccepted = 0;
}


BsJpsiPhiFilter::~BsJpsiPhiFilter()
{  
  std::cout << "Total number of accepted events = " << noAccepted << std::endl;
}

/*
HepMC::GenParticle * BsJpsiPhiFilter::findParticle(const GenPartVect genPartVect,
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

HepMC::GenParticle * BsJpsiPhiFilter::findParticle(HepMC::GenVertex* vertex, 
						   const int requested_id)
{
  for(std::vector<GenParticle*>::const_iterator p = vertex->particles_out_const_begin();
      p != vertex->particles_out_const_end(); p++)
    {
      int event_particle_id = abs( (*p)->pdg_id() );
      cout << "isC "<<event_particle_id<<"\n";
      if (requested_id == event_particle_id) return *p;
    }
  return 0;
}

HepMC::GenEvent::particle_const_iterator 
BsJpsiPhiFilter::getNextBs(const HepMC::GenEvent::particle_const_iterator start, 
			   const HepMC::GenEvent::particle_const_iterator end)
{
  HepMC::GenEvent::particle_const_iterator p;
  for (p = start; p != end; p++) 
    {
      int event_particle_id = abs( (*p)->pdg_id() );
//   cout << "search "<<event_particle_id<<"\n";
      if (event_particle_id == 531) return p;
    }
  return p;  
}


bool BsJpsiPhiFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  
  const HepMC::GenEvent * generated_event = evt->GetEvent();
  //cout << "Start\n";

  bool event_passed = false;
  HepMC::GenEvent::particle_const_iterator bs = getNextBs(generated_event->particles_begin(), 
				  		    generated_event->particles_end());
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
    cout<< "bs size "<<numChildren<<endl;
    //***
    
    /*    if ((bsChild.size()==2) && ((jpsi = findParticle(bsChild, 443))!=0) && 
	  ((phi = findParticle(bsChild, 333))!=0)) {
	  cout << bsChild[0]->momentum()<<" "<<bsChild[0]->momentum().eta()
	  <<" "<<bsChild[1]->momentum()<<" "<<bsChild[1]->momentum().eta()<<endl;
    */
    
    //***
    if( (numChildren==2) && ((jpsi = findParticle(outVertex, 443))!=0) && 
	((phi = findParticle(outVertex, 333))!=0)) {
      
      cout << jpsi->momentum().rho()<<" "<<jpsi->momentum().eta()
	   <<" "<<phi->momentum().rho() <<" "<<phi->momentum().eta()<<endl;
      cout <<"bs dec trouve"<<endl;
      if (cuts(phi, hadronCuts) && cuts(jpsi, leptonCuts)) {
        cout <<"OK trouve"<<endl;
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

/*
bool BsJpsiPhiFilter::cuts(const GenParticle * jpsi, const CutStruct cut)
{
  GenPartVect psiChild = jpsi->listChildren();
    cout << psiChild[0]->pdg_id()<<" "<<psiChild[1]->pdg_id()<<endl;
  if (psiChild.size()==2 && (abs(psiChild[0]->pdg_id()) == cut.type) &&
    (abs(psiChild[1]->pdg_id()) == cut.type))
  {
    cout << psiChild[0]->momentum()<<" "<<psiChild[0]->momentum().eta()
    <<" "<<psiChild[1]->momentum()<<" "<<psiChild[1]->momentum().eta()<<endl;
    return ( (etaInRange(psiChild[0]->momentum().eta(), cut.etaMin, cut.etaMax)) &&
    	     (etaInRange(psiChild[1]->momentum().eta(), cut.etaMin, cut.etaMax)) &&
	     (psiChild[0]->momentum().perp()> cut.ptMin) &&
	     (psiChild[1]->momentum().perp()> cut.ptMin));
  }
  return false;
}
*/

bool BsJpsiPhiFilter::cuts(const GenParticle * jpsi, const CutStruct& cut)
{
  HepMC::GenVertex* myVertex = jpsi->end_vertex();
  int numChildren = myVertex->particles_out_size();
  std::vector<HepMC::GenParticle*> psiChild;
  for(std::vector<GenParticle*>::const_iterator p = myVertex->particles_out_const_begin();
      p != myVertex->particles_out_const_end(); p++) 
    psiChild.push_back((*p));
  
  if(numChildren>1) {
    cout << psiChild[0]->pdg_id()<<" "<<psiChild[1]->pdg_id()<<endl;
  
    if (psiChild.size()==2 && (abs(psiChild[0]->pdg_id()) == cut.type) &&
	(abs(psiChild[1]->pdg_id()) == cut.type))
      {
	cout << psiChild[0]->momentum().rho()<<" "<<psiChild[0]->momentum().eta()
	     <<" "<<psiChild[1]->momentum().rho()<<" "<<psiChild[1]->momentum().eta()<<endl;
	return ( (etaInRange(psiChild[0]->momentum().eta(), cut.etaMin, cut.etaMax)) &&
		 (etaInRange(psiChild[1]->momentum().eta(), cut.etaMin, cut.etaMax)) &&
		 (psiChild[0]->momentum().perp()> cut.ptMin) &&
		 (psiChild[1]->momentum().perp()> cut.ptMin));
      }
    return false;
  }
  return false;
}

bool BsJpsiPhiFilter::etaInRange(float eta, float etamin, float etamax)
{
  return ( (etamin < eta) && (eta < etamax) );
}
