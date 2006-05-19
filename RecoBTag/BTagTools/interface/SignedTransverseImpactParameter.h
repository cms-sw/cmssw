#ifndef _BTAGGER_SIGNEDTRANSVERSEIMPACTPARAMETER_H_
#define _BTAGGER_SIGNEDTRANSVERSEIMPACTPARAMETER_H_

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include <utility>
using namespace std;
using namespace reco;


/** Transverse track impact parameter signed according to the jet 
 *  direction
 */

class SignedTransverseImpactParameter  {
  
public:

  // construct

  SignedTransverseImpactParameter(){};

  pair<bool,Measurement1D> apply(const Track &, const GlobalVector & , const Vertex &) const;

  pair<bool,Measurement1D> zImpactParameter ( const Track & , const GlobalVector & ,const Vertex & ) const ;
  
  int id() const {return 1;}

};



#endif










