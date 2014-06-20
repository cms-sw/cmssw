#ifndef ConversionVertexFinder_H
#define ConversionVertexFinder_H

/** \class ConversionVertexFinder
 *
 *
 * \author N. Marinelli - Univ. of Notre Dame
 *
 * \version   
 *
 ************************************************************/
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
//
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//
//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include <vector>

class KinematicConstrainedVertexFitter;
class ConversionVertexFinder {

public:

  ConversionVertexFinder(const edm::ParameterSet& config);


  ~ConversionVertexFinder();


  TransientVertex run (const std::vector<reco::TransientTrack>& pair);
  
  bool run( const std::vector<reco::TransientTrack>&  pair, reco::Vertex& the_vertex) ;
  

 private:
  edm::ParameterSet conf_;
  double maxDelta_; 
  double maxReducedChiSq_;
  double minChiSqImprovement_;
  int maxNbrOfIterations_;
  KinematicConstrainedVertexFitter* kcvFitter_;


};

#endif // ConversionVertexFinder_H


