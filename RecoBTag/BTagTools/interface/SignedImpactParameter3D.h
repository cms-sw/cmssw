#ifndef _BTAGGER_SIGNEDIMPACTPARAMETER3D_H_
#define _BTAGGER_SIGNEDIMPACTPARAMETER3D_H_
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include <utility>
  /** Threedimensional track impact parameter signed according to the jet 
   *  direction
   */
using namespace std;
using namespace reco;
class SignedImpactParameter3D {

public:

  // construct

  SignedImpactParameter3D(const MagneticField* field): m_field(field){};

  pair<bool,Measurement1D> apply( const Track &, const GlobalVector & direction, const Vertex & vertex) const;

  int id() const {return 2;}

  /**
   Return a pair:
   first is the decay length
   second is the distance of the track from jet axis
  */
  pair<double,Measurement1D> distanceWithJetAxis(const Track & aRecTrack, const GlobalVector & direction, const Vertex & vertex) const ;

private:

  GlobalVector distance(const TrajectoryStateOnSurface &, const Vertex &, const GlobalVector &) const;

  TrajectoryStateOnSurface closestApproachToJet(const FreeTrajectoryState &, const Vertex &, const GlobalVector &) const;

 const MagneticField* m_field;
};

#endif










