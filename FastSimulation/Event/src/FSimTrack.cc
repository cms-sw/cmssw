#include "FastSimulation/Event/interface/FSimTrack.h"

//C++ Headers
#include <iomanip>
#include <string>

//using namespace HepPDT;

FSimTrack:: FSimTrack() : 
  SimTrack(), mom_(0), id_(-1), endv_(-1),
  layer1(0), layer2(0), ecal(0), hcal(0), vfcal(0), hcalexit(0), hoentr(0), 
  prop(false), closestDaughterId_(-1), info_(0),
  properDecayTime(1E99) {;}
  
FSimTrack::FSimTrack(const RawParticle* p, 
		     int iv, int ig, int id, 
		     FBaseSimEvent* mom,
		     double dt) :
  //  SimTrack(p->pid(),*p,iv,ig),   // to uncomment once Mathcore is installed 
  SimTrack(p->pid(),p->momentum(),iv,ig), 
  mom_(mom), id_(id), endv_(-1),
  layer1(0), layer2(0), ecal(0), hcal(0), vfcal(0), hcalexit(0), hoentr(0), prop(false),
  closestDaughterId_(-1), momentum_(p->momentum()),
  properDecayTime(dt)
{ 
  setTrackId(id);
  info_ = mom_->theTable()->particle(HepPDT::ParticleID(type()));
}

FSimTrack::~FSimTrack() {;}

bool 
FSimTrack::notYetToEndVertex(const XYZTLorentzVector& pos) const {
  // If there is no end vertex, nothing to compare to
  if ( noEndVertex() ) return true;
  // If the particle immediately decays, no need to propagate
  if ( (endVertex().position()-vertex().position()).Vect().Mag2() < 1e-4 )
    return false;
  // If the end vertex has a larger radius, not yet there
  if ( endVertex().position().Perp2() > pos.Perp2()+1e-10 ) return true;
  // If the end vertex has a larger z, not yet there
  if ( fabs(endVertex().position().Z()) > fabs(pos.Z())+1e-5 ) return true;
  // Otherwise, the end vertex is overtaken already
  return false;
}

/// Set the variable at the beginning of the propagation
void 
FSimTrack::setPropagate() { 
  prop=true; 
}

/// Set the preshower layer1 variables
void 
FSimTrack::setLayer1(const RawParticle& pp, int success) { 
  Layer1_Entrance=pp; 
  layer1=success; 
}

/// Set the preshower layer2 variables
void 
FSimTrack::setLayer2(const RawParticle& pp, int success) { 
  Layer2_Entrance=pp; 
  layer2=success; 
}

/// Set the ecal variables
void 
FSimTrack::setEcal(const RawParticle& pp, int success) { 
  ECAL_Entrance=pp; 
  ecal=success; 
}

/// Set the hcal variables
void 
FSimTrack::setHcal(const RawParticle& pp, int success) { 
  HCAL_Entrance=pp; 
  hcal=success; 
}

/// Set the vcal variables
void 
FSimTrack::setVFcal(const RawParticle& pp, int success) { 
  VFCAL_Entrance=pp; 
  vfcal=success; 
}

/// Set the hcal variables
void 
FSimTrack::setHcalExit(const RawParticle& pp, int success) { 
  HCAL_Exit=pp; 
  hcalexit=success; 
}

void 
FSimTrack::setHO(const RawParticle& pp, int success) { 
  HO_Entrance=pp; 
  hoentr=success; 
}






std::ostream& operator <<(std::ostream& o , const FSimTrack& t) {

  std::string name = t.particleInfo() ? t.particleInfo()->name() : "Unknown";
  XYZTLorentzVector momentum1 = t.momentum();
  XYZVector vertex1 = t.vertex().position().Vect();
  int vertexId1 = t.vertex().id();

  o.setf(std::ios::fixed, std::ios::floatfield);
  o.setf(std::ios::right, std::ios::adjustfield);

  o << std::setw(4) << t.id() << " " 
    << std::setw(4) << t.genpartIndex() << " " 
    << name;

  for(unsigned int k=0;k<11-name.length() && k<12; k++) o << " ";  

  o << std::setw(6) << std::setprecision(2) << momentum1.eta() << " " 
    << std::setw(6) << std::setprecision(2) << momentum1.phi() << " " 
    << std::setw(6) << std::setprecision(2) << momentum1.pt() << " " 
    << std::setw(6) << std::setprecision(2) << momentum1.e() << " " 
    << std::setw(4) << vertexId1 << " " 
    << std::setw(6) << std::setprecision(1) << vertex1.x() << " " 
    << std::setw(6) << std::setprecision(1) << vertex1.y() << " " 
    << std::setw(6) << std::setprecision(1) << vertex1.z() << " "
    << std::setw(4) << t.mother().id() << " ";
  
  if ( !t.noEndVertex() ) {
    XYZTLorentzVector vertex2 = t.endVertex().position();
    int vertexId2 = t.endVertex().id();
    
    o << std::setw(4) << vertexId2 << " "
      << std::setw(6) << std::setprecision(2) << vertex2.eta() << " " 
      << std::setw(6) << std::setprecision(2) << vertex2.phi() << " " 
      << std::setw(5) << std::setprecision(1) << vertex2.pt() << " " 
      << std::setw(6) << std::setprecision(1) << vertex2.z() << " ";
    for (int i=0; i<t.nDaughters(); ++i)
      o << std::setw(4) << t.daughter(i).id() << " ";

  } else {

    if ( t.onLayer1() ) {

      XYZTLorentzVector vertex2 = t.layer1Entrance().vertex();
      
      o << std::setw(4) << -t.onLayer1() << " " 
	<< std::setw(6) << std::setprecision(2) << vertex2.eta() << " " 
	<< std::setw(6) << std::setprecision(2) << vertex2.phi() << " " 
	<< std::setw(5) << std::setprecision(1) << vertex2.pt() << " " 
	<< std::setw(6) << std::setprecision(1) << vertex2.z() << " "
	<< std::setw(6) << std::setprecision(2) << t.layer1Entrance().pt() << " " 
	<< std::setw(6) << std::setprecision(2) << t.layer1Entrance().e() << " ";
      
    } else if ( t.onEcal() ) { 

      XYZTLorentzVector vertex2 = t.ecalEntrance().vertex();
      
      o << std::setw(4) << -t.onEcal() << " " 
	<< std::setw(6) << std::setprecision(2) << vertex2.eta() << " " 
	<< std::setw(6) << std::setprecision(2) << vertex2.phi() << " " 
	<< std::setw(5) << std::setprecision(1) << vertex2.pt() << " " 
	<< std::setw(6) << std::setprecision(1) << vertex2.z() << " "
	<< std::setw(6) << std::setprecision(2) << t.ecalEntrance().pt() << " " 
	<< std::setw(6) << std::setprecision(2) << t.ecalEntrance().e() << " ";
    }
  }
  return o;
}
