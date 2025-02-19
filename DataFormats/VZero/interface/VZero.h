#ifndef _VZero_h_
#define _VZero_h_

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VZero/interface/VZeroFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <utility>

namespace reco {

  class VZeroData
  {
   public:
     float dca, impactMother, armenterosPt,armenterosAlpha;
     math::GlobalPoint crossingPoint;
     std::pair<GlobalVector,GlobalVector> momenta;
  };

  class VZero
  {
   public:
     // default constructor
     VZero() { }

     // constructor from parameters
     VZero(Vertex vertex, VZeroData data);

     // decay/conversion vertex
     Vertex vertex() const { return vertex_; }

     // position of vertex     
     Vertex::Point crossingPoint() const { return vertex_.position(); }

     // reference to positive daughter
     TrackRef positiveDaughter() const
     { return (*(vertex_.tracks_begin()  )).castTo<reco::TrackRef>(); }

     // reference to negative daughter
     TrackRef negativeDaughter() const
     { return (*(vertex_.tracks_begin()+1)).castTo<reco::TrackRef>(); }

     // distance of closest approach
     float dca() const { return data_.dca; }

     // impact parameter of the mother particle
     float impactMother() const { return data_.impactMother; }

     // Armenteros variables
     float armenterosPt()    const { return data_.armenterosPt; }
     float armenterosAlpha() const { return data_.armenterosAlpha; }

     // Momentum
     std::pair<GlobalVector,GlobalVector> momenta() const { return data_.momenta; }

   private:
     Vertex vertex_;
     VZeroData data_;
  }; 
}

#endif
