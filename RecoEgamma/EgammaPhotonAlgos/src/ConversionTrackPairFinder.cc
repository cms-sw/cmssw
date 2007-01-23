#include <iostream>
#include <vector>
#include <memory>
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackPairFinder.h"
// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
//
#include "CLHEP/Units/PhysicalConstants.h"
#include <TMath.h>
#include <vector>

//using namespace std;

ConversionTrackPairFinder::ConversionTrackPairFinder( ){ 

    LogDebug("ConversionTrackPairFinder") << " CTOR  " <<  "\n";  

}

ConversionTrackPairFinder::~ConversionTrackPairFinder() {

    LogDebug("ConversionTrackPairFinder") << " DTOR " <<  "\n";  
    
}


std::vector<std::vector<reco::TransientTrack> >  ConversionTrackPairFinder::run(std::vector<reco::TransientTrack> outInTrk,  std::vector<reco::TransientTrack> inOutTrk  ) {

  
    LogDebug("ConversionTrackPairFinder") << "::run " <<  "\n";  

  std::vector<reco::TransientTrack>  selectedOutInTk;
  std::vector<reco::TransientTrack> selectedInOutTk;
  std::vector<reco::TransientTrack> allSelectedTk;

  bool oneLeg=false;
  bool noTrack=false;  




  for( std::vector<reco::TransientTrack>::const_iterator  iTk =  outInTrk.begin(); iTk !=  outInTrk.end(); iTk++) {
      LogDebug("ConversionTrackPairFinder") << " Out In Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
   
    if ( iTk->numberOfValidHits() <3 ||   iTk->normalizedChi2() <0 ) continue; 
      selectedOutInTk.push_back(*iTk);
      allSelectedTk.push_back(*iTk);

    
  }

  for(  std::vector<reco::TransientTrack>::const_iterator  iTk =  inOutTrk.begin(); iTk !=  inOutTrk.end(); iTk++) {
      LogDebug("ConversionTrackPairFinder") << " In Out Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
   
    if ( iTk->numberOfValidHits() <3 ||   iTk->normalizedChi2() <0 ) continue; 
    selectedInOutTk.push_back(*iTk);
    allSelectedTk.push_back(*iTk);    
  }


// Sort tracks in decreasing number of hits
  if(selectedOutInTk.size() > 0)
    std::stable_sort(selectedOutInTk.begin(), selectedOutInTk.end(), ByNumOfHits());
  if(selectedInOutTk.size() > 0)
    std::stable_sort(selectedInOutTk.begin(), selectedInOutTk.end(), ByNumOfHits());
  if(allSelectedTk.size() > 0)
    std::stable_sort(allSelectedTk.begin(), allSelectedTk.end(), ByNumOfHits());



  for(  std::vector<reco::TransientTrack>::const_iterator  iTk =  selectedOutInTk.begin(); iTk !=  selectedOutInTk.end(); iTk++) {
      LogDebug("ConversionTrackPairFinder") << " Selected Out In  Tracks charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
  }
  for(  std::vector<reco::TransientTrack>::const_iterator  iTk =  selectedInOutTk.begin(); iTk !=  selectedInOutTk.end(); iTk++) {
      LogDebug("ConversionTrackPairFinder") << " Selected In Out Tracks charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
  }
  
  
  
  std::vector<reco::TransientTrack > thePair(2);
  std::vector<std::vector<reco::TransientTrack> > allPairs;
  std::vector<reco::TransientTrack>::const_iterator  iTk1;
  std::vector<reco::TransientTrack>::const_iterator  iTk2;

  if ( allSelectedTk.size()  > 2 ) {
    
    
    
    for( iTk1 =  allSelectedTk.begin(); iTk1 !=  allSelectedTk.end(); iTk1++) {
      for( iTk2 =  iTk1; iTk2 !=  allSelectedTk.end(); iTk2++) {
	if ( ( iTk1->charge() *  iTk2->charge() ) > 0 ) continue; // Reject same charge pairs
	thePair.clear();
	thePair.push_back( *iTk1 );
	thePair.push_back( *iTk2 );
	allPairs.push_back ( thePair );	
      }
    }
    
    
  }  else if (  allSelectedTk.size()  == 2 ) {
    
    if (  (allSelectedTk[0]).charge() * (allSelectedTk[1]).charge() < 0 ) {
      thePair.clear();
      thePair.push_back( allSelectedTk[0] );
      thePair.push_back( allSelectedTk[1]  );
      allPairs.push_back ( thePair );	
      
    } else {
      oneLeg=true; 
      
    }
    
  } else if  ( allSelectedTk.size()   ==1 ) { /// ONly one track in input to the finder
    oneLeg=true;  
  } else {
    noTrack=true;
  } 


  if ( oneLeg ) {
    thePair.clear();               
    iTk1 =  allSelectedTk.begin();
    thePair.push_back( (*iTk1) );
    allPairs.push_back ( thePair );
      LogDebug("ConversionTrackPairFinder") << "  WARNING ConversionTrackPairFinder::tracks The candidate has just one leg. Need to find another way to evaltuate the vertex !!! "   << "\n";
  }
  
  if ( noTrack) 
    thePair.clear();  
  
  
  return allPairs;
  
}

std::vector<std::vector<reco::Track> >  ConversionTrackPairFinder::run(const edm::Handle<reco::TrackCollection>& outInTrk, const edm::Handle<reco::TrackCollection>& inOutTrk ) {


  reco::TrackCollection selectedOutInTk;
  reco::TrackCollection selectedInOutTk;
  reco::TrackCollection allSelectedTk;

    LogDebug("ConversionTrackPairFinder") << "::run " <<  "\n";  
  for( reco::TrackCollection::const_iterator  iTk =  (*outInTrk).begin(); iTk !=  (*outInTrk).end(); iTk++) {
      LogDebug("ConversionTrackPairFinder") << " Out In Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
   
    if ( iTk->numberOfValidHits() <3 ||   iTk->normalizedChi2() <0 ) continue; 
      selectedOutInTk.push_back(*iTk);
      allSelectedTk.push_back(*iTk);

    
  }

  for( reco::TrackCollection::const_iterator  iTk =  (*inOutTrk).begin(); iTk !=  (*inOutTrk).end(); iTk++) {
      LogDebug("ConversionTrackPairFinder") << "  In Out Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
   
    if ( iTk->numberOfValidHits() <3 ||   iTk->normalizedChi2() <0 ) continue; 
    selectedInOutTk.push_back(*iTk);
    allSelectedTk.push_back(*iTk);    
  }


// Sort tracks in decreasing number of hits
  if(selectedOutInTk.size() > 0)
    std::stable_sort(selectedOutInTk.begin(), selectedOutInTk.end(), ByNumOfHits());

  if(selectedInOutTk.size() > 0)
    std::stable_sort(selectedInOutTk.begin(), selectedInOutTk.end(), ByNumOfHits());

  if(allSelectedTk.size() > 0)
    std::stable_sort(allSelectedTk.begin(), allSelectedTk.end(), ByNumOfHits());



  for( reco::TrackCollection::const_iterator  iTk =  selectedOutInTk.begin(); iTk !=  selectedOutInTk.end(); iTk++) {
      LogDebug("ConversionTrackPairFinder") << " Selected Out In  Tracks charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
  }
  for( reco::TrackCollection::const_iterator  iTk =  selectedInOutTk.begin(); iTk !=  selectedInOutTk.end(); iTk++) {
      LogDebug("ConversionTrackPairFinder") << " Selected In Out Tracks charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->innerMomentum() << "\n";  
  }


  std::vector<reco::Track > thePair(2);
  std::vector<std::vector<reco::Track> > allPairs;


    
  
    
    for( reco::TrackCollection::const_iterator  iTk1 =  allSelectedTk.begin(); iTk1 !=  allSelectedTk.end(); iTk1++) {
      for( reco::TrackCollection::const_iterator  iTk2 =  iTk1; iTk2 !=  allSelectedTk.end(); iTk2++) {
	if ( ( iTk1->charge() *  iTk2->charge() ) > 0 ) continue; // Reject same charge pairs
	thePair.clear();
	thePair.push_back( *iTk1 );
	thePair.push_back( *iTk2 );
	allPairs.push_back ( thePair );	
      }
    }
    


  return allPairs;


}

