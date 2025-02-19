// $Id: HepMCFileReader.cc,v 1.11 2009/12/01 19:23:11 fabstoec Exp $

/**  
*  See header file for a description of this class.
*
*
*  $Date: 2009/12/01 19:23:11 $
*  $Revision: 1.11 $
*  \author Jo. Weng  - CERN, Ph Division & Uni Karlsruhe
*/

#include <iostream>
#include <iomanip>
#include <string>

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/IO_GenEvent.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include "IOMC/Input/interface/HepMCFileReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;


HepMCFileReader *HepMCFileReader::instance_=0;


//-------------------------------------------------------------------------
HepMCFileReader *HepMCFileReader::instance()
{
  // Implement a HepMCFileReader singleton.

  if (instance_== 0) {
    instance_ = new HepMCFileReader();
  }
  return instance_;
}


//-------------------------------------------------------------------------
HepMCFileReader::HepMCFileReader() :
  evt_(0), input_(0)
{ 
  // Default constructor.
  if (instance_ == 0) {
    instance_ = this;  
  } else {
    edm::LogError("HepMCFileReader") << "Constructing a new instance";
  }
} 


//-------------------------------------------------------------------------
HepMCFileReader::~HepMCFileReader()
{
  edm::LogInfo("HepMCFileReader") << "Destructing HepMCFileReader";
  
  instance_=0;    
  delete input_;
}


//-------------------------------------------------------------------------
void HepMCFileReader::initialize(const string &filename)
{
  if (isInitialized()) {
    edm::LogError("HepMCFileReader") << "Was already initialized... reinitializing";
    delete input_;
  }

  edm::LogInfo("HepMCFileReader") << "Opening file" << filename << "using HepMC::IO_GenEvent";
  input_ = new HepMC::IO_GenEvent(filename.c_str(), std::ios::in);

  if (rdstate() == std::ios::failbit) {
    throw cms::Exception("FileNotFound", "HepMCFileReader::initialize()")
      << "File " << filename << " was not found.\n";
  }
}


//-------------------------------------------------------------------------
int HepMCFileReader::rdstate() const
{
  // work around a HepMC IO_ inheritence shortfall

  HepMC::IO_GenEvent *p = dynamic_cast<HepMC::IO_GenEvent*>(input_);
  if (p) return p->rdstate();

  return std::ios::failbit;
}


//-------------------------------------------------------------------------
bool HepMCFileReader::readCurrentEvent()
{
  evt_ = input_->read_next_event();
  if (evt_) { 
    edm::LogInfo("HepMCFileReader") << "| --- Event Nr. "  << evt_->event_number()
         << " with " << evt_->particles_size() << " particles --- !";  
    ReadStats();
    //  printHepMcEvent();
    //          printEvent();
  } else {
    edm::LogInfo("HepMCFileReader") << "Got no event" <<endl;
  }

  return evt_ != 0;
}


//-------------------------------------------------------------------------
bool HepMCFileReader::setEvent(int event)
{
  return true;
}


//-------------------------------------------------------------------------
bool HepMCFileReader::printHepMcEvent() const
{
  if (evt_ != 0) evt_->print(); 
  return true;
}


//-------------------------------------------------------------------------
HepMC::GenEvent * HepMCFileReader::fillCurrentEventData()
{
  readCurrentEvent();
  return evt_;
}


//-------------------------------------------------------------------------
// Print out in old CMKIN style for comparisons
void HepMCFileReader::printEvent() const {
  int mo1=0,mo2=0,da1=0,da2=0,status=0,pid=0;   
  if (evt_ != 0) {   
    cout << "---#-------pid--st---Mo1---Mo2---Da1---Da2------px------py------pz-------E-";
    cout << "------m---------x---------y---------z---------t-";
    cout << endl;
    cout.setf(ios::right, ios::adjustfield);
    for(int n=1; n<=evt_->particles_size(); n++) {  
      HepMC::GenParticle *g = index_to_particle[n];  
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
      if (g->production_vertex() != 0 && g->end_vertex() != 0 && status == 2) {
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
      cout << endl; 
    }
  } else {
    cout << " HepMCFileReader: No event available !" << endl;
  }
}


//-------------------------------------------------------------------------
void HepMCFileReader::ReadStats()
{
  unsigned int particle_counter=0;
  index_to_particle.reserve(evt_->particles_size()+1); 
  index_to_particle[0] = 0; 
  for (HepMC::GenEvent::vertex_const_iterator v = evt_->vertices_begin();
       v != evt_->vertices_end(); ++v ) {
    // making a list of incoming particles of the vertices
    // so that the mother indices in HEPEVT can be filled properly
    for (HepMC::GenVertex::particles_in_const_iterator p1 = (*v)->particles_in_const_begin();
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
    for (HepMC::GenVertex::particles_out_const_iterator p2 = (*v)->particles_out_const_begin();
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
void HepMCFileReader::getStatsFromTuple(int &mo1, int &mo2, int &da1, int &da2,
                                        int &status, int &pid, int j) const
{
  if (!evt_) {
    cout <<  "HepMCFileReader: Got no event :-(  Game over already  ?" <<endl; 
  } else {
    status =  index_to_particle[j]->status();
    pid = index_to_particle[j]->pdg_id();
    if ( index_to_particle[j]->production_vertex() ) {
      
          //HepLorentzVector p = index_to_particle[j]->
          //production_vertex()->position();
      
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
    } else {
      mo1 =0;
      mo2 =0;
    }   
    if (!index_to_particle[j]->end_vertex()) {
      //find # of 1. daughter
      int first_daughter = find_in_map( particle_to_index,
      *(index_to_particle[j]->end_vertex()->particles_begin(HepMC::children)));
      //cout <<"first_daughter "<< first_daughter <<  "num_daughters " << num_daughters << endl; 
      HepMC::GenVertex::particle_iterator ic;
      int last_daughter=0;
      //find # of last daughter
      for (ic = index_to_particle[j]->end_vertex()->particles_begin(HepMC::children);
      ic != index_to_particle[j]->end_vertex()->particles_end(HepMC::children); ++ic) 
      last_daughter= find_in_map( particle_to_index,*ic);       
      
      if (first_daughter== 0) last_daughter = 0;
      da1=first_daughter;
      da2=last_daughter;
    } else {
      da1=0;
      da2=0;
    }
  } 
} 


//-------------------------------------------------------------------------
int HepMCFileReader::find_in_map( const std::map<HepMC::GenParticle*,int>& m, 
                                  HepMC::GenParticle *p) const
{
  std::map<HepMC::GenParticle*,int>::const_iterator iter = m.find(p);
  return (iter == m.end()) ? 0 : iter->second;
}

