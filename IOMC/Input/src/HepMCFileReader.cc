/**  
*  See header file for a description of this class.
*
*
*  $Date: 2006/04/25 16:33:45 $
*  $Revision: 1.3 $
*  \author Jo. Weng  - CERN, Ph Division & Uni Karlsruhe
*/
#include "IOMC/Input/interface/HepMCFileReader.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/HepMC/ReadHepMC.h"
#include "CLHEP/HepMC/GenEvent.h"
#include<iostream>

using namespace std;
//-------------------------------------------------------------------------


HepMCFileReader * HepMCFileReader::instance_=0;
//-------------------------------------------------------------------------

HepMCFileReader * HepMCFileReader::instance(){
	
	if (instance_== 0) {
		instance_ = new HepMCFileReader();
		
	}
	return instance_;
	
}
//-------------------------------------------------------------------------


HepMCFileReader::HepMCFileReader(): initialized_(false), input_(0), index_to_particle(3996) { 
	//  if ( infoV ) cout << "Constructing a new HepMCFileReader" << endl;
	cout << "Constructing a new HepMCFileReader" << endl;
	if(instance_ == 0) instance_ = this;  
	
} 
//-------------------------------------------------------------------------


HepMCFileReader::~HepMCFileReader(){ 
	//if ( infoV ) cout << "Destructing HepMCFileReader" << endl;
	cout << "Destructing HepMCFileReader" << endl;
	
	if(input_) {
		input_->close();
		delete input_;
	}
	instance_=0;    
	
}
//-------------------------------------------------------------------------

void HepMCFileReader::setInitialized(bool value){initialized_=value;}
//-------------------------------------------------------------------------


void HepMCFileReader::initialize(const string & filename){
	
	if (initialized_) {
		cout << "HepMCFileReader was already initialized... reinitializing it " << endl;
		if(input_) {
			input_->close();
			delete input_;
		}		
	}
	cout<<"HepMCFileReader::initialize : Opening file "<<filename<<endl;	
	input_ = new ifstream(filename.c_str(), ios::in | ios::binary);
	if(! (*input_)) {
		throw cms::Exception("FileNotFound", "HepMCFileReader::initialize()")
		<< "File " << filename << " was not found.\n";
	}
       	initialized_ = true;   	
}
//-------------------------------------------------------------------------

bool HepMCFileReader::isInitialized(){
	return initialized_;
}
//-------------------------------------------------------------------------


bool  HepMCFileReader::readCurrentEvent() {
	bool filter=false;
	evt = HepMC::readGenEvent( *input_);
	if (evt){ 
		cout <<"| --- HepMCFileReader: Event Nr. "  <<evt->event_number() <<" with " <<evt->particles_size()<<" particles --- !" <<endl;	
		nParticles= evt->particles_size();
		ReadStats();
		//	printHepMcEvent();
		//          printEvent();
		filter=true;
	}
	
	if (!evt) {
		cout <<  "HepMCFileReader: Got no event :-(" <<endl;
		filter=false;
	}
	return filter;
}

//-------------------------------------------------------------------------

bool HepMCFileReader::setEvent(int event){return true;}
//-------------------------------------------------------------------------

bool HepMCFileReader::printHepMcEvent() const{
	if (evt!=0) evt->print();	
	return true;
}
//-------------------------------------------------------------------------

HepMC::GenEvent * HepMCFileReader::fillCurrentEventData(){
	readCurrentEvent();
	return evt;
}
//-------------------------------------------------------------------------
// Print out in old CMKIN style for comparisons
void HepMCFileReader::printEvent() const {
	int mo1=0,mo2=0,da1=0,da2=0,status=0,pid=0; 	
	if (evt!=0) {		
		cout << "---#-------pid--st---Mo1---Mo2---Da1---Da2------px------py------pz-------E-";
		cout << "------m---------x---------y---------z---------t-";
		cout << endl;
		cout.setf(ios::right, ios::adjustfield);
		for(int n=1; n<=evt->particles_size() ; 	n++) {  
			HepMC::GenParticle * g = index_to_particle[n];  
			getStatsFromTuple( mo1,mo2,da1,da2,status,pid,n);
			cout << setw(4) << n
			<< setw(8) << pid
			<< setw(5) << status
			<< setw(6) << mo1    
			<< setw(6) << mo2
			<< setw(6) << da1
			<< setw(6) << da2;
			cout.setf(ios::fixed, ios::floatfield);
			cout.setf(ios::right, ios::adjustfield);
			cout << setw(10) << setprecision(2) << g->momentum().x();
			cout << setw(8) << setprecision(2) << g->momentum().y();
			cout << setw(10) << setprecision(2) << g->momentum().z();
			cout << setw(8) << setprecision(2) << g->momentum().t();
			cout << setw(8) << setprecision(2) << g->generatedMass();  			
			// tau=L/(gamma*beta*c) 
			if (g->production_vertex() !=0 && g->end_vertex() != 0 && status == 2){
				cout << setw(10) << setprecision(2) <<g->production_vertex()->position().x();
				cout << setw(10) << setprecision(2) <<g->production_vertex()->position().y();
				cout << setw(10) << setprecision(2) <<g->production_vertex()->position().z();
				
				double xm = g->production_vertex()->position().x();
				double ym = g->production_vertex()->position().y();
				double zm = g->production_vertex()->position().z();
				double xd = g->end_vertex()->position().x();
				double yd = g->end_vertex()->position().y();
				double zd = g->end_vertex()->position().z();
				double decl = sqrt((xd-xm)*(xd-xm)+(yd-ym)*(yd-ym)+(zd-zm)*(zd-zm));
				double labTime = decl/c_light;
				// convert lab time to proper time
				double properTime =  labTime/g->momentum().rho()*(g->generatedMass() );
				// set the proper time in nanoseconds
				cout << setw(8) << setprecision(2) <<properTime;
			}
			else{
				cout << setw(10) << setprecision(2) << 0.0;
				cout << setw(10) << setprecision(2) << 0.0;
				cout << setw(10) << setprecision(2) << 0.0;
				cout << setw(8) << setprecision(2) << 0.0;	
			}
			cout <<"" <<endl; 
		}
	}
	else cout <<  " HepMCFileReader: No event available !" << endl;
}

//-------------------------------------------------------------------------
void HepMCFileReader::ReadStats(){
	
	unsigned int particle_counter=0;
	index_to_particle[0] = 0;	
	for (HepMC::GenEvent::vertex_const_iterator v = evt->vertices_begin();
	v != evt->vertices_end(); ++v ){
		// making a list of incoming particles of the vertices
		// so that the mother indices in HEPEVT can be filled properly
		for (HepMC::GenVertex::particles_in_const_iterator p1
		= (*v)->particles_in_const_begin();
		p1 != (*v)->particles_in_const_end(); ++p1 ) {
			++particle_counter;
			//particle_counter can be very large for heavy ions
			if(particle_counter >= index_to_particle.size() ) {
				//make it large enough to hold up to this index
				 index_to_particle.resize(particle_counter+1);
				} 					      
			index_to_particle[particle_counter] = *p1;
			particle_to_index[*p1] = particle_counter;
		}		
		// daughters are entered only if they aren't a mother of
		// another vertex
		for (HepMC::GenVertex::particles_out_const_iterator p2
		= (*v)->particles_out_const_begin();
		p2 != (*v)->particles_out_const_end(); ++p2) {
			if (!(*p2)->end_vertex()) {
				++particle_counter;
				//particle_counter can be very large for heavy ions
				if(particle_counter  >= index_to_particle.size() ) {				
				  //make it large enough to hold up to this index
				  index_to_particle.resize(particle_counter+1);
       				} 									
				index_to_particle[particle_counter] = *p2;
				particle_to_index[*p2] = particle_counter;
			}
		}
	}	
}


//-------------------------------------------------------------------------
void HepMCFileReader::getStatsFromTuple(int &mo1, int &mo2, 
int &da1, int &da2 , 
int &status, int &pid,
int j) const {
	
	if (!evt) cout <<  "HepMCFileReader: Got no event :-(  Game over already  ? " <<endl; 
	else{
		status =  index_to_particle[j]->status();
		pid = index_to_particle[j]->pdg_id();
		if ( index_to_particle[j]->production_vertex() ) {
			
			HepLorentzVector p = index_to_particle[j]->
			production_vertex()->position();
			
			int num_mothers = index_to_particle[j]->production_vertex()->
			particles_in_size();
			int first_mother = find_in_map( particle_to_index,
			*(index_to_particle[j]->
			production_vertex()->
			particles_in_const_begin()));
			int last_mother = first_mother + num_mothers - 1;
			if ( first_mother == 0 ) last_mother = 0;
			mo1=first_mother;
			mo2=last_mother;
		}
		else {
			mo1 =0;
			mo2 =0;
		}		
		if (index_to_particle[j]->hasChildren()) {
			//find # of 1. daughter
			int first_daughter = find_in_map( particle_to_index,
			*(index_to_particle[j]->end_vertex()->particles_begin(HepMC::children)));
			//cout <<"first_daughter "<< first_daughter <<  "num_daughters " << num_daughters << endl; 
			HepMC::GenVertex::particle_iterator ic;
			int last_daughter=0;
			//find # of last daughter
			for (ic  = index_to_particle[j]->end_vertex()->particles_begin(HepMC::children);
			ic != index_to_particle[j]->end_vertex()->particles_end(HepMC::children); ++ic) 
			last_daughter= find_in_map( particle_to_index,*ic); 			
			
			if ( first_daughter== 0 ) last_daughter = 0;
			da1=first_daughter;
			da2=last_daughter;
			
		}	 
		else{
			da1=0;
			da2=0;
		}
	}	
} 


//-------------------------------------------------------------------------
int HepMCFileReader::find_in_map( const std::map<HepMC::GenParticle*,int>& m, 
HepMC::GenParticle* p) const {
	std::map<HepMC::GenParticle*,int>::const_iterator iter = m.find(p);
	if ( iter == m.end() ) return 0;
	return iter->second;
}

