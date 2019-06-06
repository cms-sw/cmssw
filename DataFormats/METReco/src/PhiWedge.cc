#include "DataFormats/METReco/interface/PhiWedge.h"

/*
  [class]:  PhiWedge
  [authors]: R. Remington, The University of Florida
  [description]: See PhiWedge.h
  [date]: October 15, 2009
*/

using namespace reco;

PhiWedge::PhiWedge() {
  energy_ = 0.;
  iphi_ = 0;
  constituents_ = 0;
  OverlappingCSCTracks_ = 0;
  OverlappingCSCSegments_ = 0;
  OverlappingCSCRecHits_ = 0;
  OverlappingCSCHaloTriggers_ = 0;
  min_time_ = 0.;
  max_time_ = 0.;
  PlusZOriginConfidence_ = 0.;
}

PhiWedge::PhiWedge(float E, int iphi, int constituents) {
  energy_ = E;
  iphi_ = iphi;
  constituents_ = constituents;
  min_time_ = 0.;
  max_time_ = 0.;
  OverlappingCSCTracks_ = 0;
  OverlappingCSCSegments_ = 0;
  OverlappingCSCRecHits_ = 0;
  OverlappingCSCHaloTriggers_ = 0;
  PlusZOriginConfidence_ = 0.;
}

PhiWedge::PhiWedge(float E, int iphi, int constituents, float min_time, float max_time) {
  energy_ = E;
  iphi_ = iphi;
  min_time_ = min_time;
  max_time_ = max_time;
  constituents_ = constituents;
  OverlappingCSCTracks_ = 0;
  OverlappingCSCSegments_ = 0;
  OverlappingCSCRecHits_ = 0;
  OverlappingCSCHaloTriggers_ = 0;
  PlusZOriginConfidence_ = 0.;
}

PhiWedge::PhiWedge(const PhiWedge& wedge) {
  energy_ = wedge.Energy();
  iphi_ = wedge.iPhi();
  min_time_ = wedge.MinTime();
  max_time_ = wedge.MaxTime();
  constituents_ = wedge.NumberOfConstituents();
  OverlappingCSCTracks_ = wedge.OverlappingCSCTracks();
  OverlappingCSCHaloTriggers_ = wedge.OverlappingCSCHaloTriggers();
  OverlappingCSCRecHits_ = wedge.OverlappingCSCRecHits();
  OverlappingCSCSegments_ = wedge.OverlappingCSCSegments();
  PlusZOriginConfidence_ = wedge.PlusZOriginConfidence();
}
