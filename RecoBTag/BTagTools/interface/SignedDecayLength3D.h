#ifndef _SIGNEDDECAYLENGHT3D_H_
#define _SIGNEDDECAYLENGHT3D_H_
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include <utility>


  /** Threedimensional track decay length (minimum distance of the closest 
   *  approach to a jet from the primary vertex) signed according to the jet 
   *  direction
   */

 
class SignedDecayLength3D  {

public:

  // construct

  SignedDecayLength3D(){};

  static std::pair<bool,Measurement1D> apply(const reco::TransientTrack & track, 
                            const GlobalVector & direction, const reco::Vertex & vertex) ;


  int id() const {return 3;}
 
private:

  static TrajectoryStateOnSurface closestApproachToJet(const FreeTrajectoryState &, const reco::Vertex &, const GlobalVector &,  const MagneticField *) ;

 
};



#endif










