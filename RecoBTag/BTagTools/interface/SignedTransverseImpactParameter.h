#ifndef _BTAGGER_SIGNEDTRANSVERSEIMPACTPARAMETER_H_
#define _BTAGGER_SIGNEDTRANSVERSEIMPACTPARAMETER_H_

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include <utility>


/** Transverse track impact parameter signed according to the jet 
 *  direction
 */

class SignedTransverseImpactParameter  {
  
public:

  // construct

  SignedTransverseImpactParameter(){};

  std::pair<bool,Measurement1D> apply(const reco::TransientTrack &, const GlobalVector & , const reco::Vertex &) const;

  std::pair<bool,Measurement1D> zImpactParameter ( const reco::TransientTrack & , const GlobalVector & ,const reco::Vertex & ) const ;
  

};



#endif










