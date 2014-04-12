#ifndef TrackClassFilter_H
#define TrackClassFilter_H

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCategoryData.h"
#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"

  /**  filter to define the belonging of a track to a TrackClass
   */ 
class TrackClassFilter 
{
 public:

 TrackClassFilter() {}

 class Input
 {
 public:
  Input(const bool useQ, const reco::Track & t,const reco::Jet &j, const reco::Vertex & v) :
                     useQuality(useQ),  track(t), jet(j), vertex(v) {}
  const bool useQuality;
  const reco::Track & track;
  const reco::Jet & jet;
  const reco::Vertex & vertex;
 };

 typedef Input first_argument_type;
 typedef TrackProbabilityCalibration::Entry second_argument_type;
 typedef bool result_type;

 bool operator()(const first_argument_type & , const second_argument_type &) const;

//  void dump() const;

};


#endif








