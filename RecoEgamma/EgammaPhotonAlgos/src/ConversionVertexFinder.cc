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
  
  LogDebug("ConversionVertexFinder") << "ConversionVertexFinder CTOR  " <<  "\n";  

}

ConversionVertexFinder::~ConversionVertexFinder() {

  LogDebug("ConversionVertexFinder") << "ConversionVertexFinder DTOR " <<  "\n";  
    
}

// reco::Vertex*  ConversionVertexFinder::run(std::vector<reco::TransientTrack>  pair) {
CachingVertex  ConversionVertexFinder::run(std::vector<reco::TransientTrack>  pair) {
  LogDebug("ConversionVertexFinder") << "ConversionVertexFinder run pair size " << pair.size() <<  "\n";  
  
  for ( std::vector<reco::TransientTrack>::const_iterator iTk=pair.begin(); iTk!=pair.end(); ++iTk) {
    LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder  Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
  }
  
  
  /*
    TransientVertex* theVertex;
    try {
    
    KalmanVertexFitter myFitter(true);
    CachingVertex* cv;

    cv = new CachingVertex ( myFitter.vertex( pair)  );

    
    
    theVertex= new TransientVertex ( *cv ) ;

    
    
    } catch (std::exception & err) {
    
    LogDebug("ConversionVertexFinder")  << " ConversionVertexFinder Exception during event number: " << "\n";
    
    }
    
    
    
  if ( theVertex->isValid() ) {
  LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder VALID " << "\n";
  } else {
  LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder NOT VALID " << "\n";
  }
  
  if ( theVertex->isValid() ) {
  LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder vertex position " << theVertex->position() << "\n";
  return theVertex;
  
  } else {
  
  
  return 0;
  }
  
  */
  
  
  
  
  KalmanVertexFitter fitter;
  LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder ciao 1 " << "\n";
  CachingVertex theVertex = fitter.vertex(pair); 
  //  TransientVertex  theVertex = fitter.vertex(pair); 
  LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder ciao 2 " << "\n";
  
  if ( theVertex.isValid() ) {
    LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder VALID " << "\n";
  } else {
    LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder NOT VALID " << "\n";
  }
  
  if ( theVertex.isValid() ) {
    LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder vertex position " << theVertex.position() << "\n";
    return theVertex;
    
  }
  
  
  
  
  
}

