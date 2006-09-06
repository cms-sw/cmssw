#ifndef _VZero_h_
#define _VZero_h_

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VZero/interface/VZeroFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

namespace reco {

  class VZeroData
  {
   public:
     float dcaR,dcaZ, impactMother, armenterosPt,armenterosAlpha;
     GlobalPoint crossingPoint;
  };

  class VZero
  {
   public:
     // default construction
     VZero() { }

     // constructor from parameters
     VZero(Vertex vertex, VZeroData data);

     Vertex vertex() const { return vertex_; }
     Vertex::Point crossingPoint() const { return vertex_.position(); }

     TrackRef positiveDaughter() const { return *(vertex_.tracks_begin()  ); }
     TrackRef negativeDaughter() const { return *(vertex_.tracks_begin()+1); }

     float dcaR() const { return data_.dcaR; }
     float dcaZ() const { return data_.dcaZ; }
     float impactMother() const { return data_.impactMother; }
     float armenterosPt() const { return data_.armenterosPt; }
     float armenterosAlpha() const { return data_.armenterosAlpha; }

   private:
     Vertex vertex_;
     VZeroData data_;
  }; 
}

#endif
