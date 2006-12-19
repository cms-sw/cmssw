#include <iostream>
#include <vector>
#include <memory>
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionVertexFinder.h"
// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include "CLHEP/Units/PhysicalConstants.h"
#include <TMath.h>




ConversionVertexFinder::ConversionVertexFinder( ){ 

  std::cout << "ConversionVertexFinder CTOR  " <<  std::endl;  

}

ConversionVertexFinder::~ConversionVertexFinder() {

  std::cout << "ConversionVertexFinder DTOR " <<  std::endl;  
    
}

 reco::Vertex*  ConversionVertexFinder::run(std::vector<reco::TransientTrack>  pair) {
  std::cout << "ConversionVertexFinder run pair size " << pair.size() <<  std::endl;  

  for ( std::vector<reco::TransientTrack>::const_iterator iTk=pair.begin(); iTk!=pair.end(); ++iTk) {
    std::cout << "  ConversionVertexFinder  Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << std::endl;  
  }

  /*  
  
  KalmanVertexFitter fitter;
  TransientVertex theVertex = fitter.vertex(pair); 
  if ( theVertex.isValid() ) {
    std::cout << "  ConversionVertexFinder VALID " << std::endl;
  }

  if ( theVertex.isValid() ) {
    std::cout << "  ConversionVertexFinder vertex position " << theVertex.position() << std::endl;
    return &theVertex;

  } else {
     
    return &theVertex;
  }

  */


 }

