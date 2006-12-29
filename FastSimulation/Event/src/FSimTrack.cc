//#include "CommonReco/PatternTools/interface/TTrack.h"

#include "CLHEP/HepMC/GenParticle.h"
#include "FastSimulation/Event/interface/FBaseSimEvent.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Event/interface/FSimTrack.h"

//C++ Headers
#include <iomanip>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace HepPDT;

FSimTrack:: FSimTrack() : 
  SimTrack(), mom_(0), id_(-1), endv_(-1),
  layer1(0), layer2(0), ecal(0), hcal(0), vfcal(0), prop(false) {;}
  
//FSimTrack::FSimTrack(int embd, FBaseSimEvent* mom) : 
//  mom_(mom), embd_(embd), id_(mom->nTracks()), endv_(-1),
//  layer1(0), layer2(0), ecal(0), hcal(0), vfcal(0), prop(false) {;}
  
FSimTrack::FSimTrack(const RawParticle* p, int iv, int ig, int id, FBaseSimEvent* mom) :
  SimTrack(p->pid(),*p,iv,ig), mom_(mom), id_(id), endv_(-1),
  layer1(0), layer2(0), ecal(0), hcal(0), vfcal(0), prop(false) { setTrackId(id);}

FSimTrack::~FSimTrack() {
  // Clear the maps 
  //  theRecHits.clear();
  //  theSimHits.clear();
}

const DefaultConfig::ParticleData*
FSimTrack::particleInfo() const {
  return mom_->theTable()->particle(ParticleID(type()));
}

//float 
//FSimTrack::charge() const { return particleInfo()->charge();}
  
const FSimVertex& 
FSimTrack::vertex() const { return mom_->vertex(vertIndex()); }

const FSimVertex& 
FSimTrack::endVertex() const { return mom_->vertex(endv_); }

const FSimTrack& 
FSimTrack::mother() const { return vertex().parent(); }

int
FSimTrack::nDaughters() const { return abs(type()) != 11 ? 
				  endVertex().nDaughters() : 
                                  daugh_.size(); }

const FSimTrack& 
FSimTrack::daughter(int i) const { return abs(type()) != 11 ?
				     endVertex().daughter(i) : 
                                     mom_->track(daugh_[i]); }

const vector<int>&
FSimTrack::daughters() const { return abs(type()) != 11 ? 
				 endVertex().daughters() : 
                                 daugh_; }

bool  
FSimTrack::noEndVertex() const { 
  return 
    // The particle either has no end vertex index
    endv_ == -1 || 
    // or it's an electron that has just brem'ed, but continues its way
    ( abs(type())==11 && 
      endVertex().nDaughters()>0 && 
      endVertex().daughter(endVertex().nDaughters()-1).type()==22); } 

bool 
FSimTrack::notYetToEndVertex(const HepLorentzVector& pos) const {
  // If there is no end vertex, nothing to compare to
  if ( noEndVertex() ) return true;
  // If the particle immediately decays, no need to propagate
  if ( (endVertex().position()-vertex().position()).vect().mag() < 0.01 )
    return false;
  // If the end vertex has a larger radius, not yet there
  if ( endVertex().position().perp() > pos.perp()+0.00001 ) return true;
  // If the end vertex has a larger z, not yet there
  if ( fabs(endVertex().position().z()) > fabs(pos.z())+0.00001 ) return true;
  // Otherwise, the end vertex is overtaken already
  return false;
}

bool  
FSimTrack::noMother() const { return noVertex() || vertex().noParent(); }

bool  
FSimTrack::noDaughter() const { return noEndVertex() || !nDaughters(); }

const HepMC::GenParticle* 
FSimTrack::genParticle() const { return mom_->embdGenpart(genpartIndex()); }
   
//const SimTrack& 
//FSimTrack::me() const { return mom_->embdTrack(embd_); } 

/*
double
FSimTrack::distFromRecTrack(const TTrack& theRecTrack) const {


  // The hits of the reconstructed track
  vector<RecHit> theRecHits = theRecTrack.recHits();

  // The innermost three hits (tracks are not reconstructed with less 
  // than three hits)
  vector<GlobalPoint> firstRecHits;
  firstRecHits.push_back(theRecHits[0].globalPosition());
  firstRecHits.push_back(theRecHits[1].globalPosition());
  firstRecHits.push_back(theRecHits[2].globalPosition());

  double totDist = 0.;
  // Check if the simulated track shares these innermost hits
  for ( unsigned hit=0; hit<3; ++hit ) {
    double dmin = 1000000.;
    // Loop over the tracker layers
    for ( unsigned layer=1; layer<28; ++layer ) {
      if ( !isARecHit(layer) ) continue;
      GlobalPoint firstSimHit = recHit(layer)->globalPosition();
      double dist = (firstSimHit-firstRecHits[hit]).mag2(); 
      if ( dist < dmin ) dmin = dist;
    }
    totDist += dmin;
  }

  totDist = sqrt(totDist);

  return totDist;

}
*/

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

/// Set the hcal variables
void 
FSimTrack::setVFcal(const RawParticle& pp, int success) { 
  VFCAL_Entrance=pp; 
  vfcal=success; 
}

/*
/// Add a RecHit on a tracker layer
void 
FSimTrack::addRecHit(const FamosBasicRecHit* hit, unsigned layer) { 
  // theRecHits[layer]=hit;
  theRecHits.insert(make_pair(layer,hit));
}
*/

/// Add a SimHit on a tracker layer
/*
void 
FSimTrack::addSimHit(const RawParticle& pp, unsigned layer) { 
  //    theSimHits[layer]=pp; 
  if ( theSimHits.size() == 0 ) mom_->addChargedTrack(id_);
  theSimHits.insert(pair<unsigned,RawParticle>(layer,pp));

}
*/
    
/*
/// Is there a RecHit on this layer?
bool 
FSimTrack::isARecHit(const unsigned layer) const { 
  return ( recHits().find(layer) != recHits().end() ); }

/// If yes, here it is.
static const FamosBasicRecHit* zeroHit;
const FamosBasicRecHit*
FSimTrack::recHit(unsigned layer) const {
  if ( isARecHit(layer) ) return recHits().find(layer)->second;
  return zeroHit;
}
*/

/*
/// Is there a SimHit on this layer?
bool 
FSimTrack::isASimHit(const unsigned layer) const { 
  return ( simHits().find(layer) != simHits().end() ); }

/// If yes, here is the corresponding RawParticle
static const RawParticle zeroTrack;
const RawParticle& 
FSimTrack::simHit(unsigned layer) const {
  if ( isASimHit(layer) ) return theSimHits.find(layer)->second;
  return zeroTrack;
}
*/

ostream& operator <<(ostream& o , const FSimTrack& t) {

  string name = t.particleInfo()->name();
  HepLorentzVector momentum1 = t.momentum();
  Hep3Vector vertex1 = t.vertex().position().vect();
  int vertexId1 = t.vertex().id();

  o.setf(ios::fixed, ios::floatfield);
  o.setf(ios::right, ios::adjustfield);

  o << setw(4) << t.id() << " " 
    << setw(4) << t.genpartIndex() << " " 
    << name;

  for(unsigned int k=0;k<11-name.length() && k<12; k++) o << " ";  

  o << setw(6) << setprecision(2) << momentum1.eta() << " " 
    << setw(6) << setprecision(2) << momentum1.phi() << " " 
    << setw(6) << setprecision(2) << momentum1.perp() << " " 
    << setw(6) << setprecision(2) << momentum1.e() << " " 
    << setw(4) << vertexId1 << " " 
    << setw(6) << setprecision(1) << vertex1.x() << " " 
    << setw(6) << setprecision(1) << vertex1.y() << " " 
    << setw(6) << setprecision(1) << vertex1.z() << " "
    << setw(4) << t.mother().id() << " ";
  
  if ( !t.noEndVertex() ) {
    HepLorentzVector vertex2 = t.endVertex().position();
    int vertexId2 = t.endVertex().id();
    
    o << setw(4) << vertexId2 << " "
      << setw(6) << setprecision(2) << vertex2.eta() << " " 
      << setw(6) << setprecision(2) << vertex2.phi() << " " 
      << setw(5) << setprecision(1) << vertex2.perp() << " " 
      << setw(6) << setprecision(1) << vertex2.z() << " ";
    for (int i=0; i<t.nDaughters(); ++i)
      o << setw(4) << t.daughter(i).id() << " ";

  } else {

    if ( t.onLayer1() ) {

      HepLorentzVector vertex2 = t.layer1Entrance().vertex();
      
      o << setw(4) << -t.onLayer1() << " " 
	<< setw(6) << setprecision(2) << vertex2.eta() << " " 
	<< setw(6) << setprecision(2) << vertex2.phi() << " " 
	<< setw(5) << setprecision(1) << vertex2.perp() << " " 
	<< setw(6) << setprecision(1) << vertex2.z() << " "
	<< setw(6) << setprecision(2) << t.layer1Entrance().perp() << " " 
	<< setw(6) << setprecision(2) << t.layer1Entrance().e() << " ";
      
    } else if ( t.onEcal() ) { 

      HepLorentzVector vertex2 = t.ecalEntrance().vertex();
      
      o << setw(4) << -t.onEcal() << " " 
	<< setw(6) << setprecision(2) << vertex2.eta() << " " 
	<< setw(6) << setprecision(2) << vertex2.phi() << " " 
	<< setw(5) << setprecision(1) << vertex2.perp() << " " 
	<< setw(6) << setprecision(1) << vertex2.z() << " "
	<< setw(6) << setprecision(2) << t.ecalEntrance().perp() << " " 
	<< setw(6) << setprecision(2) << t.ecalEntrance().e() << " ";
    }
  }
  return o;
}
