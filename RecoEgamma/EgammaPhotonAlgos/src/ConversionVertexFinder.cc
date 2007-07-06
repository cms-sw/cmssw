#include <iostream>
#include <vector>
#include <memory>
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionVertexFinder.h"
// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
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


CachingVertex  ConversionVertexFinder::run(std::vector<reco::TransientTrack>  pair) {
  LogDebug("ConversionVertexFinder") << "ConversionVertexFinder run pair size " << pair.size() <<  "\n";  
  
  //for ( std::vector<reco::TransientTrack>::const_iterator iTk=pair.begin(); iTk!=pair.end(); ++iTk) {
  // LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder  Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
  //}
  
  
  
  KalmanVertexFitter fitter;
  CachingVertex theVertex;
  const string metname =  "ConversionVertexFinder| ConversionVertexFinder";
  try{
    theVertex = fitter.vertex(pair); 
  }  catch ( cms::Exception& e ) {
    // std::cout << " cms::Exception caught in ConversionVertexFinder::run " << "\n" ;
    edm::LogWarning(metname) << "cms::Exception caught in ConversionVertexFinder::run\n"
			     << e.explainSelf();
    
  }
  

  if ( theVertex.isValid() ) {
    LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder VALID " << "\n";
    LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder vertex position " << theVertex.position() << "\n"; 
  } else {
    LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder NOT VALID " << "\n";
  }
  
  return theVertex;
    
    
}

