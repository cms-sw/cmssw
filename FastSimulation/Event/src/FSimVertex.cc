#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"

static const FSimTrack pTrack;
const FSimTrack&
FSimVertex::parent() const {
  if ( noParent() ) return pTrack;
  int id = me()->mother()->barcode()-1;
  return mom_->track(id);
}

static const FSimTrack d1Track;
const FSimTrack&
FSimVertex::daughter1() const {
  if ( noDaughter() ) return d1Track;
  int id = me()->listChildren().front()->barcode()-1;
  return mom_->track(id);
}

static const FSimTrack d2Track;
const FSimTrack&
FSimVertex::daughter2() const {
  if ( noDaughter() ) return d2Track;
  int id = me()->listChildren().back()->barcode()-1;
  return mom_->track(id);
}

std::ostream& operator <<(std::ostream& o , const FSimVertex& t) {
  return o << t.me();
}
