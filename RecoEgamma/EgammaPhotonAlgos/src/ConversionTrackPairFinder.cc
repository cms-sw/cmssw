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
#include "DataFormats/EgammaTrackReco/interface/TrackSuperClusterAssociation.h"
//
#include "CLHEP/Units/PhysicalConstants.h"
#include <TMath.h>
#include <vector>
#include <map>

//using namespace std;

ConversionTrackPairFinder::ConversionTrackPairFinder( ){ 

    LogDebug("ConversionTrackPairFinder") << " CTOR  " <<  "\n";  

}

ConversionTrackPairFinder::~ConversionTrackPairFinder() {

    LogDebug("ConversionTrackPairFinder") << " DTOR " <<  "\n";  
    
}




 std::map<std::vector<reco::TransientTrack>, reco::SuperCluster>  ConversionTrackPairFinder::run(std::vector<reco::TransientTrack> outInTrk,  
										const edm::Handle<reco::TrackCollection>& outInTrkHandle,
										const edm::Handle<reco::TrackSuperClusterAssociationCollection>& outInTrackSCAssH, 
										std::vector<reco::TransientTrack> inOutTrk, 
										const edm::Handle<reco::TrackCollection>& inOutTrkHandle,
										const edm::Handle<reco::TrackSuperClusterAssociationCollection>& inOutTrackSCAssH  ) {


  
  
  //  LogDebug("ConversionTrackPairFinder") << "::run " <<  "\n";  
  LogDebug("ConversionTrackPairFinder")  << "ConversionTrackPairFinder::run " <<  "\n";  
  
  std::vector<reco::TransientTrack>  selectedOutInTk;
  std::vector<reco::TransientTrack> selectedInOutTk;
  std::vector<reco::TransientTrack> allSelectedTk;
  std::map<reco::TransientTrack, reco::SuperCluster>  scTrkAssocMap; 

  bool oneLeg=false;
  bool noTrack=false;  
  

  
  int iTrk=0;
  for( std::vector<reco::TransientTrack>::const_iterator  iTk =  outInTrk.begin(); iTk !=  outInTrk.end(); iTk++) {
    LogDebug("ConversionTrackPairFinder")  << " Out In Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum()  << "\n";  
    
    if ( iTk->numberOfValidHits() <3 ||   iTk->normalizedChi2() <0 ) continue; 
    
    
    
    const reco::TrackTransientTrack* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
    reco::TrackRef myTkRef= ttt->persistentTrackRef(); 
    LogDebug("ConversionTrackPairFinder") <<  " ConversionTrackPairFinder persistent track ref hits " << myTkRef->recHitsSize() << " inner momentum " <<  myTkRef->innerMomentum() << "\n";
    
    edm::Ref<reco::TrackCollection> trackRef(outInTrkHandle, iTrk );
    
    LogDebug("ConversionTrackPairFinder") <<  " ConversionTrackPairFinder track from handle hits " << trackRef->recHitsSize() << " inner momentum " <<  trackRef->innerMomentum() << "\n";
    
    reco::TrackSuperClusterAssociationCollection outInTrackSCAss = *outInTrackSCAssH;
    const reco::SuperCluster aClus= *outInTrackSCAss[trackRef];
    
    LogDebug("ConversionTrackPairFinder") << "ConversionTrackPairFinder  Out In track belonging to SC with energy " << aClus.energy() << "\n"; 

    scTrkAssocMap[*iTk]= aClus;
    selectedOutInTk.push_back(*iTk);
    allSelectedTk.push_back(*iTk);

    iTrk++;
    
  }


  iTrk=0;
  for(  std::vector<reco::TransientTrack>::const_iterator  iTk =  inOutTrk.begin(); iTk !=  inOutTrk.end(); iTk++) {
    LogDebug("ConversionTrackPairFinder") << " In Out Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
    
    if ( iTk->numberOfValidHits() <3 ||   iTk->normalizedChi2() <0 ) continue; 
    
    
    const reco::TrackTransientTrack* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
    reco::TrackRef myTkRef= ttt->persistentTrackRef(); 
    LogDebug("ConversionTrackPairFinder") <<  " ConversionTrackPairFinder persistent track ref hits " << myTkRef->recHitsSize() << " inner momentum " <<  myTkRef->innerMomentum() << "\n";
    
    edm::Ref<reco::TrackCollection> trackRef(inOutTrkHandle, iTrk );
    
    LogDebug("ConversionTrackPairFinder") <<  " ConversionTrackPairFinder track from handle hits " << trackRef->recHitsSize() << " inner momentum " <<  trackRef->innerMomentum() << "\n";
    
    reco::TrackSuperClusterAssociationCollection inOutTrackSCAss = *inOutTrackSCAssH;
    const reco::SuperCluster aClus= *inOutTrackSCAss[trackRef];
    
    scTrkAssocMap[*iTk]= aClus;
    selectedInOutTk.push_back(*iTk);
    allSelectedTk.push_back(*iTk);    

    iTrk++;
    
  }
  

  LogDebug("ConversionTrackPairFinder") << " ConversionTrackPairFinder allSelectedTk size " << allSelectedTk.size() << "  scTrkAssocMap  size " <<  scTrkAssocMap.size() << "\n"; 
  
  // Sort tracks in decreasing number of hits
  if(selectedOutInTk.size() > 0)
    std::stable_sort(selectedOutInTk.begin(), selectedOutInTk.end(), ByNumOfHits());
  if(selectedInOutTk.size() > 0)
    std::stable_sort(selectedInOutTk.begin(), selectedInOutTk.end(), ByNumOfHits());
  if(allSelectedTk.size() > 0)
    std::stable_sort(allSelectedTk.begin(), allSelectedTk.end(), ByNumOfHits());
  
  
  
  for(  std::vector<reco::TransientTrack>::const_iterator  iTk =  selectedOutInTk.begin(); iTk !=  selectedOutInTk.end(); iTk++) {
    LogDebug("ConversionTrackPairFinder") << " Selected Out In  Tracks charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
  }
  for(  std::vector<reco::TransientTrack>::const_iterator  iTk =  selectedInOutTk.begin(); iTk !=  selectedInOutTk.end(); iTk++) {
    LogDebug("ConversionTrackPairFinder") << " Selected In Out Tracks charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
  }
  
  
  
  std::vector<reco::TransientTrack > thePair(2);
  std::vector<std::vector<reco::TransientTrack> > allPairs;
  std::map<std::vector<reco::TransientTrack>, reco::SuperCluster> allPairSCAss;

  std::map<reco::TransientTrack, reco::SuperCluster>::const_iterator iMap1;
  std::map<reco::TransientTrack, reco::SuperCluster>::const_iterator iMap2;

  
  if ( scTrkAssocMap.size() > 2 ){
    for( iMap1 =   scTrkAssocMap.begin(); iMap1 !=   scTrkAssocMap.end(); ++iMap1) {
      for( iMap2 =  iMap1; iMap2 !=   scTrkAssocMap.end(); ++iMap2) {
	if (   ((iMap1->first)).charge() *  ((iMap2->first)).charge()  > 0 ) continue;


	if( !( (  fabs( ((iMap1->second)).energy() -((iMap2->second)).energy() ) < 0.001 ) &&  
	       (   fabs( ((iMap1->second)).eta() -   ((iMap2->second)).eta() ) < 0.001 )      &&
	       (    fabs( ((iMap1->second)).phi() - ((iMap2->second)).phi() ) < 0.001  ) ) )   continue;
	


	

	LogDebug("ConversionTrackPairFinder") << " ConversionTrackPairFinde All selected from the map First  Track charge " <<   (iMap1->first).charge() << " Num of RecHits " <<  ((iMap1->first)).recHitsSize() << " inner momentum " <<  ((iMap1->first)).track().innerMomentum() << " Ass SC " << (iMap1->second).energy() <<  "\n";  
	
	LogDebug("ConversionTrackPairFinder") << " ConversionTrackPairFinde All selected from the map Second  Track charge " <<   ((iMap2->first)).charge() << " Num of RecHits " <<  ((iMap2->first)).recHitsSize() << " inner momentum " <<  ((iMap2->first)).track().innerMomentum() << " Ass SC " << (iMap2->second).energy()  <<  "\n";  
	
	
	thePair.clear();
	thePair.push_back( iMap1->first );
	thePair.push_back( iMap2->first  );
	allPairs.push_back ( thePair );	
	
	allPairSCAss[thePair]= iMap1->second; 
	
	
      }
    }



  } else if (  (scTrkAssocMap.size() ==2) ) {

    iMap1=scTrkAssocMap.begin();
    iMap2=scTrkAssocMap.end();
    if (  ((iMap1->second).energy()== (iMap2->second).energy()  )   && ((iMap1->first).charge() * (iMap2->first).charge() < 0 )   ) {
      
      thePair.clear();
      thePair.push_back( iMap1->first );
      thePair.push_back( iMap2->first );
      allPairs.push_back ( thePair );	

      allPairSCAss[thePair]= iMap1->second; 
      
    } else {
      oneLeg=true; 
      
    }
    
  } else if  (scTrkAssocMap.size() ==1   ) { /// ONly one track in input to the finder
    oneLeg=true;  
  } else {
    noTrack=true;
  } 
  
  
  if ( oneLeg ) {
    thePair.clear();               


    iMap1=scTrkAssocMap.begin();   
    thePair.push_back(iMap1->first);
 
    allPairs.push_back ( thePair );
    allPairSCAss[thePair]= iMap1->second; 

    LogDebug("ConversionTrackPairFinder") << "  WARNING ConversionTrackPairFinder::tracks The candidate has just one leg. Need to find another way to evaltuate the vertex !!! "   << "\n";
  }
  
  if ( noTrack) {
    thePair.clear();  
    allPairSCAss.clear();
  }
  
  //  return allPairs;

  return allPairSCAss;
 

}


