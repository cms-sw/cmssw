#include "GeneratorInterface/GenFilters/interface/JetFlavourCutFilter.h"

using namespace edm;
using namespace std;
using namespace HepMC;



JetFlavourCutFilter::JetFlavourCutFilter(const edm::ParameterSet& iConfig)
{
  token_ = consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")));
  jetType = iConfig.getParameter< int >("jetType");
  if ((jetType>=1)&&(jetType<=3)) jetType=1;

  noAccepted = 0;
}


JetFlavourCutFilter::~JetFlavourCutFilter()
{  
  std::cout << "Total number of accepted events = " << noAccepted << std::endl;
}


HepMC::GenParticle * JetFlavourCutFilter::findParticle(const GenPartVect& genPartVect,
	const int requested_id)
{
  for (GenPartVectIt p = genPartVect.begin(); p != genPartVect.end(); p++)
    {
      if (requested_id == (*p)->pdg_id()) return *p;
    }
  return 0;
}



void
JetFlavourCutFilter::printHisto(const HepMC::GenEvent::particle_iterator start, 
			       const HepMC::GenEvent::particle_iterator end)
{
  HepMC::GenEvent::particle_iterator p;
  for (p = start; p != end; p++) 
    {
      //vector< GenParticle * > parents = (*p)->listParents();
      vector< GenParticle * > parents;
      HepMC::GenVertex* inVertex = (*p)->production_vertex();
      for(std::vector<GenParticle*>::const_iterator iter = inVertex->particles_in_const_begin();
	  iter != inVertex->particles_in_const_end();iter++)
	parents.push_back(*iter);
      
      cout << "isC "<<(*p)->pdg_id()<<" status "<<(*p)->status()<<" Parents: "<<parents.size()<<" - ";
      for (GenPartVectIt z = parents.begin(); z != parents.end(); z++){
	cout << (*z)->pdg_id()<<" ";
      }

      //vector< GenParticle * > child = (*p)->listChildren();
      vector< GenParticle * > child;
      HepMC::GenVertex* outVertex = (*p)->end_vertex();
      for(std::vector<GenParticle*>::const_iterator iter = outVertex->particles_in_const_begin();
	  iter != outVertex->particles_in_const_end();iter++)
	child.push_back(*iter);
      
      cout << " - Child: "<<child.size()<<" - ";
      for (GenPartVectIt z = child.begin(); z != child.end(); z++){
	cout << (*z)->pdg_id()<<" ";
      }
      cout<<"\n";
    }
}


bool JetFlavourCutFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  
  const HepMC::GenEvent * generated_event = evt->GetEvent();

  bool event_passed = false;

  vector<int> foundQ;
  HepMC::GenEvent::particle_const_iterator p;
  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); p++) {
    if (((*p)->pdg_id() < 1) && ((*p)->pdg_id() > -10)) { // We have an anti-quark
//       cout << "antiQ "<< (*p)->pdg_id();
      //vector< GenParticle * > parents = (*p)->listParents();
      vector< GenParticle * > parents;
      HepMC::GenVertex* inVertex = (*p)->production_vertex();
      for(std::vector<GenParticle*>::const_iterator iter = inVertex->particles_in_const_begin();
	  iter != inVertex->particles_in_const_end();iter++)
	parents.push_back(*iter);
      
      for (GenPartVectIt z = parents.begin(); z != parents.end(); z++){

	vector< GenParticle * > child;
	HepMC::GenVertex* outVertex = (*z)->end_vertex();
	for(std::vector<GenParticle*>::const_iterator iter = outVertex->particles_in_const_begin();
	    iter != outVertex->particles_in_const_end();iter++)
	  child.push_back(*iter);
	
	if (findParticle(child, -(*p)->pdg_id())) foundQ.push_back(-(*p)->pdg_id());
      }
//       cout << " "<< foundQ.size()<<endl;
      
    }

  }
  
  int flavour = 0;
  int ff[6];
  ff[0]=0; ff[1]=0; ff[2]=0; ff[3]=0; ff[4]=0; ff[5]=0; 
  for (vector<int>::iterator i = foundQ.begin(); i != foundQ.end(); i++){
    ++ff[(*i)-1];
    if ((*i)>flavour) flavour = (*i);
  }
  // Is it light quark jet?
  if ((flavour>=0)&&(flavour<=3)) flavour=1;
  // Do we have more than one heavy flavour ?
  if ( (ff[3] && ff[4]) || (ff[3] && ff[5]) || (ff[4] && ff[5]) ) flavour =0;

  if (jetType!=flavour) event_passed=true;

//   cout <<"Final flavour: " << ff[0]<<" "<<ff[1]<<" "<<ff[2]<<" "<<ff[3]<<" "<<ff[4]<<" "<<ff[5];
//   if (ff[0]||ff[1]||ff[2])  cout << " light";
//   if (ff[3])  cout << " charm";
//   if (ff[4])  cout << " bottom";
//   if (ff[5])  cout << " top";
//   if ((ff[0]||ff[1]||ff[2])&&((ff[3] + ff[4]+ff[5])))  {cout << " LH";};
//   if ( (ff[3] && ff[4]) || (ff[3] && ff[5]) || (ff[4] && ff[5]) ) {cout <<" notDef";}//printHisto(start,end);//}

//   cout<<" "<< flavour << event_passed << endl;

  if (event_passed) noAccepted++;

  delete generated_event; 

  return event_passed;
}
