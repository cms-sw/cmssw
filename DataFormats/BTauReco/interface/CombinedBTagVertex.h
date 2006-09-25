#ifndef BTauRecoCombinedBTagVertexh
#define BTauRecoCombinedBTagVertexh

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"

namespace reco {
  class CombinedBTagVertex  {
  public:
    /**
     *   A data class used for storing b-tagging info
     *   that can be associated with a vertex
     */

    CombinedBTagVertex(); 
    CombinedBTagVertex( const reco::Vertex & vertex,
        const GlobalVector & trackVector, double mass,
        bool isV0, double fracPV,
        const Measurement1D & flightdistance_2d = Measurement1D(),
        const Measurement1D & flightdistance_3d = Measurement1D() );

    void setFlightDistance2D ( const Measurement1D & );
    void setFlightDistance3D ( const Measurement1D & );

    void print() const;

    double chi2() const;
    double ndof() const;
    int nTracks() const;
    double mass() const;
    const reco::Vertex & vertex() const;
    const GlobalVector & trackVector() const;
    bool isV0() const;
    double fracPV() const;
    Measurement1D flightDistance2D() const;
    Measurement1D flightDistance3D() const;
    bool isValid() const;

  private:
    reco::Vertex vertex_;
    GlobalVector trackVector_;  // sum of all tracks at this vertex
    double mass_;       /** mass computed from all charged tracks at this
                          * vertex assuming Pion mass hypothesis.
                          * For now, loop over all tracks and
                          * compute m^2 = Sum(E^2) - Sum(p^2)
                          */
    bool   isV0_;        // has been tagged as V0 (true) or not (false);
    double fracPV_;      // fraction of tracks also used to build primary vertex

    Measurement1D d2_; //< flight distance, 2d
    Measurement1D d3_; //< flight distance, 3d
    bool isValid_;
  };
}

#endif
