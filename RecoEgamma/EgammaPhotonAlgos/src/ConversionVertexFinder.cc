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
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include <TMath.h>





ConversionVertexFinder::ConversionVertexFinder( ){ 
  
  LogDebug("ConversionVertexFinder") << "ConversionVertexFinder CTOR  " <<  "\n";  

}

ConversionVertexFinder::~ConversionVertexFinder() {

  LogDebug("ConversionVertexFinder") << "ConversionVertexFinder DTOR " <<  "\n";  
    
}


TransientVertex  ConversionVertexFinder::run(std::vector<reco::TransientTrack>  pair) {
  LogDebug("ConversionVertexFinder") << "ConversionVertexFinder run pair size " << pair.size() <<  "\n";  
  
  //for ( std::vector<reco::TransientTrack>::const_iterator iTk=pair.begin(); iTk!=pair.end(); ++iTk) {
  // LogDebug("ConversionVertexFinder") << "  ConversionVertexFinder  Tracks in the pair  charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
  //}


  reco::Vertex theVertex;  
  KalmanVertexFitter fitter(true);
  TransientVertex transientVtx;

  const string metname =  "ConversionVertexFinder| ConversionVertexFinder";
  try{

    transientVtx = fitter.vertex(pair); 

  }  catch ( cms::Exception& e ) {


    edm::LogWarning(metname) << "cms::Exception caught in ConversionVertexFinder::run\n"
			     << e.explainSelf();
    
  }
  

  return transientVtx;

    
    
}

