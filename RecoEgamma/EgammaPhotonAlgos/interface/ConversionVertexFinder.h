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
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//
//Kinematic constraint vertex fitter
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include <RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h>
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/ColinearityKinematicConstraint.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include <vector>


class ConversionVertexFinder {

public:

  ConversionVertexFinder();


  ~ConversionVertexFinder();


  TransientVertex run (std::vector<reco::TransientTrack> pair);
  
  bool run(std::vector<reco::TransientTrack>  pair, reco::Vertex& the_vertex) ;
  



};

#endif // ConversionVertexFinder_H


