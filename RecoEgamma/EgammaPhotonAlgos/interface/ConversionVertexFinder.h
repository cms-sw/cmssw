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
//
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include <vector>


class ConversionVertexFinder {

public:

  ConversionVertexFinder();


  ~ConversionVertexFinder();


 reco::Vertex* run (std::vector<reco::TransientTrack> pair);





};

#endif // ConversionVertexFinder_H


