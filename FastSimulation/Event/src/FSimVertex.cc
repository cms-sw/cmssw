#include "FastSimulation/Event/interface/FBaseSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"

  /// Default constructor
FSimVertex::FSimVertex() : SimVertex(), mom_(0), id_(-1) {;}
  
  /// constructor from the embedded vertex index in the FBaseSimEvent
FSimVertex::FSimVertex(const HepLorentzVector& v, int im, int id, FBaseSimEvent* mom) : 
  // When SimVertices were in mm (until 110_pre2)
  //    SimVertex(v.vect()*10.,v.e()*10.,im), mom_(mom), id_(id) {;}
  // Now, SimVertices are in cm (finally!)
    SimVertex(v.vect(),v.e(),im), mom_(mom), id_(id) {;}

const FSimTrack&
FSimVertex::parent() const { return mom_->track(parentIndex()); }

const FSimTrack&
FSimVertex::daughter(int i) const { return mom_->track(daugh_[i]); }

//const SimVertex& 
//FSimVertex::me() const { return mom_->embdVertex(embd_); } 

std::ostream& operator <<(std::ostream& o , const FSimVertex& t) {
  return o << t;
}
