#include "FastSimulation/Event/interface/FBaseSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"

  /// Default constructor
FSimVertex::FSimVertex() : mom_(0), embd_(-1), id_(-1) {;}
  
  /// constructor from the embedded vertex index in the FBaseSimEvent
FSimVertex::FSimVertex(int embd, FBaseSimEvent* mom) : 
    mom_(mom), embd_(embd), id_(mom->nVertices()) {;}

const FSimTrack&
FSimVertex::parent() const { return mom_->track(me().parentIndex()); }

const FSimTrack&
FSimVertex::daughter(int i) const { return mom_->track(daugh_[i]); }

const EmbdSimVertex& 
FSimVertex::me() const { return mom_->embdVertex(embd_); } 

std::ostream& operator <<(std::ostream& o , const FSimVertex& t) {
  return o << t.me();
}
