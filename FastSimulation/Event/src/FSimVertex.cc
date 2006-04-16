#include "FastSimulation/Event/interface/FBaseSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"

  /// Default constructor
FSimVertex::FSimVertex() : EmbdSimVertex(), mom_(0), id_(-1) {;}
  
  /// constructor from the embedded vertex index in the FBaseSimEvent
FSimVertex::FSimVertex(const HepLorentzVector& v, int im, int id, FBaseSimEvent* mom) : 
    EmbdSimVertex(v.vect(),v.e(),im), mom_(mom), id_(id) {;}

const FSimTrack&
FSimVertex::parent() const { return mom_->track(parentIndex()); }

const FSimTrack&
FSimVertex::daughter(int i) const { return mom_->track(daugh_[i]); }

//const EmbdSimVertex& 
//FSimVertex::me() const { return mom_->embdVertex(embd_); } 

std::ostream& operator <<(std::ostream& o , const FSimVertex& t) {
  return o << t;
}
