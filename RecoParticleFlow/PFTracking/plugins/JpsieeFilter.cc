#include "JpsieeFilter.h"
#include <algorithm>

using namespace edm;
using namespace std;
using namespace HepMC;



JpsieeFilter::JpsieeFilter(const edm::ParameterSet& iConfig)
{
  label_ = iConfig.getUntrackedParameter("moduleLabel",std::string("source"));
  motherId = iConfig.getParameter<std::vector<int> > ("motherId");
  leptonCuts.type = iConfig.getParameter< int >("leptonType");
  leptonCuts.etaMin = iConfig.getParameter<double>("leptonEtaMin");
  leptonCuts.etaMax = iConfig.getParameter<double>("leptonEtaMax");
  leptonCuts.ptMin = iConfig.getParameter<double>("leptonPtMin");

  noAccepted = 0;
}


JpsieeFilter::~JpsieeFilter()
{  
  std::cout << "Total number of accepted events = " << noAccepted << std::endl;
}

HepMC::GenEvent::particle_const_iterator 
JpsieeFilter::getNextParticle(const HepMC::GenEvent::particle_const_iterator start, 
			      const HepMC::GenEvent::particle_const_iterator end)
{
  HepMC::GenEvent::particle_const_iterator p;
  unsigned nmothers=motherId.size();
  //  std::cout << " Starting getNextParticle"  << std::endl;
  for (p = start; p != end; p++) 
    {
      int event_particle_id = abs( (*p)->pdg_id() );
//   cout << "search "<<event_particle_id<<"\n";
//      std::cout << " Current part " << event_particle_id << std::endl;
      bool result=false;
      for(unsigned ic=0;ic<nmothers;++ic)
	{
	  if(event_particle_id==motherId[ic])
	    {
	      result=true;
	      //	      std::cout << " Found it " << std::endl;
	      return p;
	      break;
	    }
	}
    }
  return p;  
}


bool JpsieeFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(label_, evt);
  
  const HepMC::GenEvent * generated_event = evt->GetEvent();
  //  generated_event->print();
  
  bool event_passed = false;
  HepMC::GenEvent::particle_const_iterator part = getNextParticle(generated_event->particles_begin(), 
								  generated_event->particles_end());
  if(part==generated_event->particles_end()) return false;
  
  while(part!=generated_event->particles_end())
    {
      HepMC::GenVertex* outVertex = (*part)->end_vertex();
      int numChildren = outVertex->particles_out_size();
      //      std::cout << " Found a " << (*part)->pdg_id() << " " << numChildren << std::endl;
      if( numChildren==2) 
	{
	  bool foundpos=false;
	  bool foundneg=false;
	  for(std::set<GenParticle*>::const_iterator p = outVertex->particles_out_const_begin();
	      p != outVertex->particles_out_const_end(); p++)
	    {
	      int event_particle_id =  (*p)->pdg_id() ;
	      //	      std::cout << " Son part id " << event_particle_id << std::endl;
	      float eta =(*p)->momentum().eta();
	      float pt =(*p)->momentum().perp();
	      if(event_particle_id==leptonCuts.type &&
		 eta > leptonCuts.etaMin &&
		 eta < leptonCuts.etaMax &&
	     pt > leptonCuts.ptMin)
		foundneg=true;
	  
	      if(event_particle_id==-leptonCuts.type &&
		 eta > leptonCuts.etaMin &&
		 eta < leptonCuts.etaMax &&
		 pt > leptonCuts.ptMin)
		foundpos=true;	  
	    }
	  if(foundneg&&foundpos)
	    {	  
	      //	      std::cout << " Good - goes into an electron pair" << " " << iEvent.id() << std::endl;
	      event_passed = true;
	      break;
	    }
	}
      part = getNextParticle(++part,generated_event->particles_end());
    }  
  
  if (event_passed) noAccepted++;
  //  cout << "End filter\n";
  
  
  return event_passed;
}

