/**  
*  $Date:  1.2006
*  \author Joanna Weng  - CERN, Ph Division & Uni Karlsruhe
*/
#include "IOMC/NtupleConverter/interface/Ntuple2HepMCFiller.h"

#include<iostream>
#include "IOMC/NtupleConverter/interface/NtupleROOTFile.h"  
using namespace std;
using namespace HepMC;

Ntuple2HepMCFiller * Ntuple2HepMCFiller::instance_=0;
//-------------------------------------------------------------------------
Ntuple2HepMCFiller * Ntuple2HepMCFiller::instance(){	
	
	if (instance_== 0) {
		instance_ = new Ntuple2HepMCFiller();
	}
	return instance_;
}


Ntuple2HepMCFiller::Ntuple2HepMCFiller(): 
   initialized_(false), input_(0),
   index_to_particle(3996),evtid(1),ntpl_id(1) { 
	cout << "Constructing a new Ntuple2HepMCFiller" << endl;
	if(instance_ == 0) instance_ = this;  	
} 

//-------------------------------------------------------------------------
Ntuple2HepMCFiller::~Ntuple2HepMCFiller(){ 
	cout << "Destructing Ntuple2HepMCFiller" << endl;	
	if(input_) {
		delete input_;
	}
	instance_=0;    
}
//-------------------------------------------------------------------------

void Ntuple2HepMCFiller::setInitialized(bool value){initialized_=value;}

//-------------------------------------------------------------------------
void Ntuple2HepMCFiller::initialize(const string & filename, int id){
	
	if (initialized_) {
		cout << "Ntuple2HepMCFiller was already initialized... reinitializing it " << endl;
		if(input_) {
			delete input_;
		}		
	}       
	
	cout<<"Ntuple2HepMCFiller::initialize : Opening file "<<filename<<endl;
	ntpl_id=id;
	input_ = new NtupleROOTFile( filename, id);
	initialized_ = true;   
}

//-------------------------------------------------------------------------

bool Ntuple2HepMCFiller::isInitialized(){
	
	return initialized_;
}

//-------------------------------------------------------------------------
bool  Ntuple2HepMCFiller::readCurrentEvent() {
	bool filter=false;
	// 1. create an empty event container
	HepMC::GenEvent* event = new HepMC::GenEvent();
	input_ ->setEvent(evtid);
	// 2. fill the evt container - if the read is successful, set the pointer
	if ( this->toGenEvent( evtid, event ) ) evt=event;
	if (evt){ 
		cout <<"| --- Ntuple2HepMCFiller: Event Nr. "  <<evt->event_number() <<" with " <<evt->particles_size()<<" particles --- !" <<endl;		
		//	printHepMcEvent();	
		filter=true;
	}
	if (!evt) {
		cout <<  "Ntuple2HepMCFiller: Got no event :-(" <<endl;
		filter=false;
		delete  evt;  // @@@
	}	
	if (evtid > input_->getNevhep()) filter = false; 
	evtid++;
	return filter;
}

//-------------------------------------------------------------------------
bool Ntuple2HepMCFiller::setEvent(unsigned int event){
  evtid= event;
  return true;
}

//-------------------------------------------------------------------------

bool Ntuple2HepMCFiller::printHepMcEvent() const{
	if (evt!=0) evt->print();	
	return true;
}

//-------------------------------------------------------------------------
HepMC::GenEvent * Ntuple2HepMCFiller::fillCurrentEventData(){
  if (readCurrentEvent())return evt;
  else return NULL;
  
}


//-------------------------------------------------------------------
bool Ntuple2HepMCFiller::toGenEvent( int evtnum, HepMC::GenEvent* evt ){
        // Written according to HepMC /fio/ IO_HEPEVT.cc( hepmc-01-26 tag at Savannah)
	// 1. set event number
	// evt->set_event_number( input_->getNevhep());
	evt->set_event_number( evtnum ) ; 
	// these do not have the correct defaults if hepev4 is not filled
	//evt->set_event_scale( hepev4()->scalelh[1] );
	//evt->set_alphaQCD( hepev4()->alphaqcdlh );
	//evt->set_alphaQED( hepev4()->alphaqedlh ); 
	std::vector<HepMC::GenParticle*> hepevt_particle(input_->getNhep()+1 );
	//2. create a particle instance for each HEPEVT entry and fill a map
	//    create a vector which maps from the HEPEVT particle index to the 
	//    GenParticle address
	//    (+1 in size accounts for hepevt_particle[0] which is unfilled)
	hepevt_particle[0] = 0;
	for ( int i1 = 1; i1 <= input_->getNhep(); ++i1 ) {
		hepevt_particle[i1] = createParticle(i1);
	}	
	std::set<HepMC::GenVertex*> new_vertices; 
	
	// 3.+4. loop over HEPEVT particles AGAIN, this time creating vertices
	for ( int i = 1; i <= input_->getNhep(); ++i ) {
		// We go through and build EITHER the production or decay 
		// vertex for each entry in hepevt
		// Note: since the HEPEVT pointers are bi-directional, it is
		///      sufficient to do one OR the other 
	  	buildProductionVertex( i, hepevt_particle, evt, 1 );
	}
	//	hepevt_particle.clear();
	return true;
}

//-------------------------------------------------------
HepMC::GenParticle* Ntuple2HepMCFiller::createParticle( int index ) {
	
	// Builds a particle object corresponding to index in HEPEVT
	HepMC::GenParticle* p 
	= new HepMC::GenParticle(FourVector( input_->getPhep(index,0), 
	input_->getPhep(index,1), 
	input_->getPhep(index,2), 
	input_->getPhep(index,3)),
	input_->getIdhep(index), 
	input_->getIsthep(index));
	p->setGeneratedMass( input_->getPhep(index, 4) );
	p->suggest_barcode( index );
	return p;
}

//-------------------------------------------------------
void Ntuple2HepMCFiller::buildProductionVertex( int i, 
std::vector<HepMC::GenParticle*>& hepevt_particle, 
HepMC::GenEvent* evt, bool printInconsistencyErrors )
{
	// 
	// for particle in HEPEVT with index i, build a production vertex
	// if appropriate, and add that vertex to the event
	HepMC::GenParticle* p = hepevt_particle[i];
	// a. search to see if a production vertex already exists
	int mother = input_->getJmohep(i,0);
	// this is dangerous.  Copying null pointer and attempting to fill 
	//  information in prod_vtx......
	HepMC::GenVertex* prod_vtx = p->production_vertex();
	//  no production vertex and mother exist -> produce vertex
	while ( !prod_vtx && mother > 0 ) {
		prod_vtx = hepevt_particle[mother]->end_vertex();
		if ( prod_vtx ) prod_vtx->add_particle_out( p );
		// increment mother for next iteration
		if ( ++mother > input_->getJmohep(i,1) ) mother = 0;
	}
	// b. if no suitable production vertex exists - and the particle
	// has atleast one mother or position information to store - 
	// make one@@@
	FourVector prod_pos( input_->getVhep(i,0), input_->getVhep(i,1), 
	input_->getVhep(i,2), input_->getVhep(i,3));
       	// orginal:
	if ( !prod_vtx && (number_parents(i)>0	|| prod_pos!=FourVector(0,0,0,0) )){
		prod_vtx = new HepMC::GenVertex();	
		prod_vtx->add_particle_out( p );
		evt->add_vertex( prod_vtx );
	}
	// c. if prod_vtx doesn't already have position specified, fill it
	if ( prod_vtx && prod_vtx->position()==FourVector(0,0,0,0) ) {
		prod_vtx->set_position( prod_pos );
	}
	// d. loop over mothers to make sure their end_vertices are
	//     consistent
	mother = input_->getJmohep(i,0);
	while ( prod_vtx && mother > 0 ) {
		if ( !hepevt_particle[mother]->end_vertex() ) {
			// if end vertex of the mother isn't specified, do it now
			prod_vtx->add_particle_in( hepevt_particle[mother] );
		} 
		else if (hepevt_particle[mother]->end_vertex() != prod_vtx ) {
			// problem scenario --- the mother already has a decay
			// vertex which differs from the daughter's produciton 
			// vertex. This means there is internal
			// inconsistency in the HEPEVT event record. Print an
			// error
			// Note: we could provide a fix by joining the two 
			//       vertices with a dummy particle if the problem
			//       arrises often with any particular generator.
			if ( printInconsistencyErrors ) std::cerr
				<< "Ntuple2HepMCFiller:: inconsistent mother/daughter "
			<< "information in HEPEVT event " 
			<< input_->getNevhep()
			<< std::endl;
		}
		if ( ++mother > input_->getJmohep(i,1) ) mother = 0;
	}
}


//-------------------------------------------------------
int Ntuple2HepMCFiller::number_children( int index )
{
	int firstchild = input_->getJdahep(index,0);
	return ( firstchild>0 ) ? 
	( 1+input_->getJdahep(index,1)-firstchild ) : 0;
}

//-------------------------------------------------------	
int Ntuple2HepMCFiller::number_parents( int index ) {

	// Fix for cmkin single particle ntpls 
	if (input_->getNhep()==1) return 1;
	// Fix for cmkindouble particle ntpls
	if (input_->getNhep()==2) return 1;	
	int firstparent = input_->getJmohep(index,0);
	if( firstparent <= 0 ) return 0;
	int secondparent = input_->getJmohep(index,1);
	if( secondparent <= 0 ) return 1;
	return secondparent - firstparent + 1;
	
} 

/*************************************************************************************
NEVHEP = event number
NHEP   = number of entries (particles, partons)
ISTHEP = status code
IDHEP  = PDG identifier
JMOHEP = position of 1st and 2nd  mother
JDAHEP = position of 1st and last daughter
PHEP   = 4-momentum and mass            (single precision in ntuple file)
VHEP   = vertex xyz and production time (single precision in ntuple file)


IRNMCP	    = run number
IEVMCP         = event number
WGTMCP         = event weight
XSECN          = cross section equivalent
IFILTER        = filter pattern
NVRMCP         = number of additional variables
VARMCP(NMXMCP) = list of additional variables
Note: VARMCP(1) = PARI(17) (=s_hat) is reserved.

There are the following default values:

IRNMCP = 1
IEVMCP = NEVHEP
WGTMCP = 1.0
XSECN  = IFILTER = 0
NVRMCP = 1
*****************************************************************************************/
