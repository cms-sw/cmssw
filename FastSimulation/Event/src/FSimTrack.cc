//#include "CommonReco/PatternTools/interface/TTrack.h"

#include "CLHEP/HepMC/GenParticle.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Event/interface/FSimTrack.h"

#include "SimGeneral/HepPDT/interface/HepPDTable.h"
#include "SimGeneral/HepPDT/interface/HepParticleData.h"

//C++ Headers
#include <iomanip>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

FSimTrack::~FSimTrack() {
  // Clear the maps 
  //  theRecHits.clear();
  theSimHits.clear();
}

const HepParticleData*
FSimTrack::particleInfo() const {
  return HepPDT::theTable().getParticleData(type());
}

float 
FSimTrack::charge() const { return particleInfo()->charge();}
  
static  const FSimVertex oVertex;
const FSimVertex& 
FSimTrack::vertex() const {
  if ( noVertex() ) return oVertex;
  int id = -me()->production_vertex()->barcode()+1;
  return mom_->vertex(id);
}

static const FSimVertex eVertex;
const FSimVertex& 
FSimTrack::endVertex() const { 
  if ( noEndVertex() ) return eVertex;
  int id = -me()->end_vertex()->barcode()+1;
  return mom_->vertex(id);
}

static const FSimTrack mTrack;
const FSimTrack& 
FSimTrack::mother() const { 
  if ( noMother() ) return mTrack;
  int id = me()->mother()->barcode()-1;
  return mom_->track(id);
}

static const FSimTrack d1Track;
const FSimTrack& 
FSimTrack::daughter1() const { 
  if ( noDaughter() ) return d1Track;
  int id = me()->listChildren().front()->barcode()-1;
  return mom_->track(id);
}

static const FSimTrack d2Track;
const FSimTrack& 
FSimTrack::daughter2() const { 
  if ( noDaughter() ) return d2Track;
  int id = me()->listChildren().back()->barcode()-1;
  return mom_->track(id);
}

bool 
FSimTrack::notYetToEndVertex(const HepLorentzVector& pos) const {
  // If there is no end vertex, nothing to compare to
  if ( noEndVertex() ) return true;
  // If the particle immediately decays, no need to propagate
  if ( (endVertex().position()-vertex().position()).vect().mag() < 0.1 ) 
    return false;
  // If the end vertex has a larger radius, not yet there
  if ( endVertex().position().perp() > pos.perp()+0.0001 ) return true;
  // If the end vertex has a larger z, not yet there
  if ( fabs(endVertex().position().z()) > fabs(pos.z())+0.0001 ) return true;
  // Otherwise, the end vertex is overtaken already
  return false;
}

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
void 
FSimTrack::addSimHit(const RawParticle& pp, unsigned layer) { 
  //    theSimHits[layer]=pp; 
  theSimHits.insert(pair<unsigned,RawParticle>(layer,pp));
}
    
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

  for(unsigned int k=0;k<9-name.length() && k<10; k++) o << " ";  

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
      << setw(6) << setprecision(1) << vertex2.z() << " "
      << setw(4) << t.daughter1().id() << " "
      << setw(4) << t.daughter2().id() << " ";

  } else {

    if ( t.onLayer1() ) {

      HepLorentzVector vertex2 = t.layer1Entrance().vertex()*0.1;
      
      o << setw(4) << -t.onLayer1() << " " 
	<< setw(6) << setprecision(2) << vertex2.eta() << " " 
	<< setw(6) << setprecision(2) << vertex2.phi() << " " 
	<< setw(5) << setprecision(1) << vertex2.perp() << " " 
	<< setw(6) << setprecision(1) << vertex2.z() << " "
	<< setw(6) << setprecision(2) << t.layer1Entrance().perp() << " " 
	<< setw(6) << setprecision(2) << t.layer1Entrance().e() << " ";
      
    } else if ( t.onEcal() ) { 

      HepLorentzVector vertex2 = t.ecalEntrance().vertex()*0.1;
      
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
