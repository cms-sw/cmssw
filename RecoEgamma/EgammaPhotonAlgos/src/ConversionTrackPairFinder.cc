#include <iostream>
#include <vector>
#include <memory>
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionTrackPairFinder.h"
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//

//
#include <vector>
#include <map>


//using namespace std;

ConversionTrackPairFinder::ConversionTrackPairFinder( ){ 

  LogDebug("ConversionTrackPairFinder") << " CTOR  " <<  "\n";  

}

ConversionTrackPairFinder::~ConversionTrackPairFinder() {

  LogDebug("ConversionTrackPairFinder") << " DTOR " <<  "\n";  
    
}




 std::map<std::vector<reco::TransientTrack>,  reco::CaloClusterPtr, CompareTwoTracksVectors>  ConversionTrackPairFinder::run(const std::vector<reco::TransientTrack>& outInTrk,  
													const edm::Handle<reco::TrackCollection>& outInTrkHandle,
													const edm::Handle<reco::TrackCaloClusterPtrAssociation>& outInTrackSCAssH, 
													const std::vector<reco::TransientTrack>& _inOutTrk, 
													const edm::Handle<reco::TrackCollection>& inOutTrkHandle,
													const edm::Handle<reco::TrackCaloClusterPtrAssociation>& inOutTrackSCAssH  ) 
{
  std::vector<reco::TransientTrack> inOutTrk = _inOutTrk;

  LogDebug("ConversionTrackPairFinder")  << "ConversionTrackPairFinder::run " <<  "\n";  
  
  std::vector<reco::TransientTrack>  selectedOutInTk;
  std::vector<reco::TransientTrack>  selectedInOutTk;
  std::vector<reco::TransientTrack>  allSelectedTk;
  std::map<reco::TransientTrack,  reco::CaloClusterPtr,CompareTwoTracks> scTrkAssocMap; 
  std::multimap<int,reco::TransientTrack,std::greater<int> >  auxMap; 
 
  bool oneLeg=false;
  bool noTrack=false;  
  
  
  
  int iTrk=0;
  for( std::vector<reco::TransientTrack>::const_iterator  iTk =  outInTrk.begin(); iTk !=  outInTrk.end(); iTk++) {
    edm::Ref<reco::TrackCollection> trackRef(outInTrkHandle, iTrk );    
    iTrk++;
    
    if ( iTk->numberOfValidHits() <3 ||   iTk->normalizedChi2() > 5000 ) continue; 
    if ( fabs(iTk->impactPointState().globalPosition().x()) > 110 ||
         fabs(iTk->impactPointState().globalPosition().y()) > 110 ||
	 fabs(iTk->impactPointState().globalPosition().z()) > 280 ) continue;
    
    //    std::cout  << " Out In Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner pt  " << sqrt(iTk->track().innerMomentum().perp2()) << "\n";  
    const reco::TrackTransientTrack* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
    reco::TrackRef myTkRef= ttt->persistentTrackRef(); 
    //std::cout <<  " ConversionTrackPairFinder persistent track ref hits " << myTkRef->recHitsSize() << " inner pt  " << sqrt(iTk->track().innerMomentum().perp2()) << "\n";  
    //    std::cout <<  " ConversionTrackPairFinder track from handle hits " << trackRef->recHitsSize() << " inner pt  " << sqrt(iTk->track().innerMomentum().perp2()) << "\n";  

    const reco::CaloClusterPtr  aClus = (*outInTrackSCAssH)[trackRef];

    //    std::cout << "ConversionTrackPairFinder  Reading the OutIn Map  " << *outInTrackSCAss[trackRef] <<  " " << &outInTrackSCAss[trackRef] <<  std::endl;
    //    std::cout << "ConversionTrackPairFinder  Out In track belonging to SC with energy " << aClus->energy() << "\n"; 

    int nHits=iTk->recHitsSize();
    scTrkAssocMap[*iTk]= aClus;
    auxMap.insert(std::pair<int,reco::TransientTrack >(nHits,(*iTk)) );
    selectedOutInTk.push_back(*iTk);
    allSelectedTk.push_back(*iTk);


    
  }


  iTrk=0;
  for(  std::vector<reco::TransientTrack>::const_iterator  iTk =  inOutTrk.begin(); iTk !=  inOutTrk.end(); iTk++) {
    edm::Ref<reco::TrackCollection> trackRef(inOutTrkHandle, iTrk );
    iTrk++;
    
    if ( iTk->numberOfValidHits() <3 ||   iTk->normalizedChi2() >5000 ) continue; 
    if ( fabs(iTk->impactPointState().globalPosition().x()) > 110 ||
         fabs(iTk->impactPointState().globalPosition().y()) > 110 ||
	 fabs(iTk->impactPointState().globalPosition().z()) > 280 ) continue;
    
    //    std::cout << " In Out Track charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner pt  " << sqrt(iTk->track().innerMomentum().perp2()) << "\n";   
    const reco::TrackTransientTrack* ttt = dynamic_cast<const reco::TrackTransientTrack*>(iTk->basicTransientTrack());
    reco::TrackRef myTkRef= ttt->persistentTrackRef(); 
    // std::cout <<  " ConversionTrackPairFinder persistent track ref hits " << myTkRef->recHitsSize() << " inner pt  " << sqrt(iTk->track().innerMomentum().perp2()) << "\n";  
    //    std::cout <<  " ConversionTrackPairFinder track from handle hits " << trackRef->recHitsSize() << " inner pt  " << sqrt(iTk->track().innerMomentum().perp2()) << "\n";  
    
    const reco::CaloClusterPtr  aClus = (*inOutTrackSCAssH)[trackRef];

    //    std::cout << "ConversionTrackPairFinder  Filling the InOut Map  " << &(*inOutTrackSCAss[trackRef]) << " " << &inOutTrackSCAss[trackRef] <<  std::endl;
    // std::cout << "ConversionTrackPairFinder  In Out  track belonging to SC with energy " << aClus.energy() << "\n"; 
    
    scTrkAssocMap[*iTk]= aClus;
    int nHits=iTk->recHitsSize();
    auxMap.insert(std::pair<int,reco::TransientTrack >(nHits,(*iTk)) );
    selectedInOutTk.push_back(*iTk);
    allSelectedTk.push_back(*iTk);    


    
  }
  

  //  std::cout << " ConversionTrackPairFinder allSelectedTk size " << allSelectedTk.size() << "  scTrkAssocMap  size " <<  scTrkAssocMap.size() << "\n"; 
  
  // Sort tracks in decreasing number of hits
  if(selectedOutInTk.size() > 0)
    std::stable_sort(selectedOutInTk.begin(), selectedOutInTk.end(), ByNumOfHits());
  if(selectedInOutTk.size() > 0)
    std::stable_sort(selectedInOutTk.begin(), selectedInOutTk.end(), ByNumOfHits());
  if(allSelectedTk.size() > 0)
    std::stable_sort(allSelectedTk.begin(),   allSelectedTk.end(),   ByNumOfHits());
  
  
  
  //  for(  std::vector<reco::TransientTrack>::const_iterator  iTk =  selectedOutInTk.begin(); iTk !=  selectedOutInTk.end(); iTk++) {
    // std::cout << " Selected Out In  Tracks charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
  //}
  //  for(  std::vector<reco::TransientTrack>::const_iterator  iTk =  selectedInOutTk.begin(); iTk !=  selectedInOutTk.end(); iTk++) {
  // std::cout << " Selected In Out Tracks charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " inner momentum " << iTk->track().innerMomentum() << "\n";  
  // }
  //  for(  std::vector<reco::TransientTrack>::const_iterator  iTk =  allSelectedTk.begin(); iTk !=  allSelectedTk.end(); iTk++) {
  // std::cout << " All Selected  Tracks charge " << iTk->charge() << " Num of RecHits " << iTk->recHitsSize() << " chi2 " << iTk->normalizedChi2() <<  " pt " <<  sqrt(iTk->track().innerMomentum().perp2()) << "\n";  
  //}
  

  
  
  std::vector<reco::TransientTrack > thePair(2);
  std::vector<std::vector<reco::TransientTrack> > allPairs;

  std::map<std::vector<reco::TransientTrack>,  reco::CaloClusterPtr, CompareTwoTracksVectors > allPairSCAss;
  std::map<std::vector<reco::TransientTrack>,  reco::CaloClusterPtr, CompareTwoTracksVectors> allPairOrdInPtSCAss;
  std::map<reco::TransientTrack,  reco::CaloClusterPtr>::const_iterator iMap1;
  std::map<reco::TransientTrack,  reco::CaloClusterPtr>::const_iterator iMap2;

 for( iMap1 =   scTrkAssocMap.begin(); iMap1 !=   scTrkAssocMap.end(); ++iMap1) {
 
   //   std::cout << " Ass map track  charge " << (iMap1->first).charge()  <<" pt " << sqrt(((iMap1->first)).track().innerMomentum().Perp2()) << " SC E  " << (iMap1->second)->energy() << " SC eta " << (iMap1->second)->eta() <<  " SC phi " << (iMap1->second)->phi() << std::endl;
  }


 std::multimap<int, reco::TransientTrack>::const_iterator iAux;


 // for( iAux = auxMap.begin(); iAux!= auxMap.end(); ++iAux) {
 //  //   std::cout << " Aux Map  " << (iAux->first)  <<" pt " << sqrt(((iAux->second)).track().innerMomentum().Perp2()) << std::endl;
 // for( iMap1 =   scTrkAssocMap.begin(); iMap1 !=   scTrkAssocMap.end(); ++iMap1) {
 //   if ( (iMap1->first) == (iAux->second) ) std::cout << " ass SC " <<  (iMap1->second)->energy() << std::endl;
 // }
 // }

  if ( scTrkAssocMap.size() > 2 ){
    

    for( iMap1 =   scTrkAssocMap.begin(); iMap1 !=   scTrkAssocMap.end(); ++iMap1) {
      for( iMap2 =  iMap1; iMap2 !=   scTrkAssocMap.end(); ++iMap2) {
    	// consider only tracks associated to the same SC 

        if (  (iMap1->second) != (iMap2->second) ) continue;  
	
	if (   ((iMap1->first)).charge() *  ((iMap2->first)).charge()  < 0 ) {
	  
	  //	  std::cout << " ConversionTrackPairFinde All selected from the map First  Track charge " <<   (iMap1->first).charge() << " Num of RecHits " <<  ((iMap1->first)).recHitsSize() << " inner pt " <<  sqrt(((iMap1->first)).track().innerMomentum().Perp2()) << " Ass SC " << (iMap1->second)->energy() <<  "\n";  
	  
	  //  std::cout << " ConversionTrackPairFinde All selected from the map Second  Track charge " <<   ((iMap2->first)).charge() << " Num of RecHits " <<  ((iMap2->first)).recHitsSize() << " inner pt " <<  sqrt(((iMap2->first)).track().innerMomentum().Perp2()) << " Ass SC " << (iMap2->second)->energy()  <<  "\n";  



	  thePair.clear();
	  thePair.push_back( iMap1->first );
	  thePair.push_back( iMap2->first  );
	  allPairs.push_back ( thePair );	
	  allPairSCAss[thePair]= iMap1->second; 

	}
      }
    }

    //    std::cout << " ConversionTrackPairFinder  INTERMIDIATE allPairSCAss size " << allPairSCAss.size() << "\n";

    if ( allPairSCAss.size() == 0) { 
      //      std::cout << " All Tracks had the same charge: Need to send out a single track  " <<   "\n";

      for( iMap1 =   scTrkAssocMap.begin(); iMap1 !=   scTrkAssocMap.end(); ++iMap1) {

	thePair.clear();
	thePair.push_back(iMap1->first);
	allPairs.push_back ( thePair );
	allPairSCAss[thePair]= iMap1->second; 
	
      }

    }





  } else if (  (scTrkAssocMap.size() ==2) ) {
    
    iMap1=scTrkAssocMap.begin();//get the first
    iMap2=iMap1;
    iMap2++;//get the second
    if (  (iMap1->second) == (iMap2->second)  ) {
      if  ( (iMap1->first).charge() * (iMap2->first).charge() < 0 )  {
	
	//	std::cout << " ConversionTrackPairFinder Case when  (scTrkAssocMap.size() ==2)  " <<   (iMap1->first).charge() << std::endl;
	//std::cout << " Num of RecHits " <<  ((iMap1->first)).recHitsSize() << std::endl;
	//	std::cout << " inner pt " <<  sqrt(((iMap1->first)).track().innerMomentum().Perp2()) << std::endl; 
	//std::cout << " Ass SC " << (iMap1->second)->energy() <<  "\n";  

	//	std::cout << " ConversionTrackPairFinder Case when  (scTrkAssocMap.size() ==2)  " <<   (iMap2->first).charge() << std::endl;
	//	std::cout << " Num of RecHits " <<  ((iMap2->first)).recHitsSize() << std::endl;
	//std::cout << " inner pt " <<  sqrt(((iMap2->first)).track().innerMomentum().Perp2()) << std::endl; 
	//std::cout << " Ass SC " << (iMap2->second)->energy() <<  "\n";  
	
	thePair.clear();
	thePair.push_back( iMap1->first );
	thePair.push_back( iMap2->first );
	allPairs.push_back ( thePair );	
	
	allPairSCAss[thePair]= iMap1->second; 


      } else {
	//std::cout << " ConversionTrackPairFinder oneLeg case when 2 tracks with same sign Pick up the longest one" << std::endl;
	if (  ((iMap1->first)).recHitsSize() > ((iMap2->first)).recHitsSize() ) {
	  thePair.clear();
	  thePair.push_back(iMap1->first);
	  allPairs.push_back ( thePair );
	  allPairSCAss[thePair]= iMap1->second; 
	} else {
	  thePair.clear();
	  thePair.push_back(iMap2->first);
	  allPairs.push_back ( thePair );
	  allPairSCAss[thePair]= iMap2->second;
	}
      }
      
    }

  } else if  (scTrkAssocMap.size() ==1   ) { /// ONly one track in input to the finder
    //    std::cout << " ConversionTrackPairFinder oneLeg case when 1 track only " << std::endl;
    oneLeg=true;  
  } else {
    noTrack=true;
  } 
  
  
  if ( oneLeg ) {
    thePair.clear();               
    // std::cout << " ConversionTrackPairFinder oneLeg case charge  " << std::endl;
						      
						      
    iMap1=scTrkAssocMap.begin();   
						      
    //std::cout << " ConversionTrackPairFinder oneLeg case charge  " <<   (iMap1->first).charge() << " Num of RecHits " <<  ((iMap1->first)).recHitsSize() << " inner pt " <<  sqrt(((iMap1->first)).track().innerMomentum().Perp2()) << " Ass SC " << (iMap1->second)->energy() <<  "\n";  
						      
    thePair.push_back(iMap1->first);
    allPairs.push_back ( thePair );
    allPairSCAss[thePair]= iMap1->second; 

    // std::cout << "  WARNING ConversionTrackPairFinder::tracks The candidate has just one leg. Need to find another way to evaltuate the vertex !!! "   << "\n";
  }
  
  if ( noTrack) {
    //    std::cout << "  WARNING ConversionTrackPairFinder::tracks case noTrack "   << "\n";  
    thePair.clear();  
    allPairSCAss.clear();
  }
  


  /// all cases above failed and some track-SC association is still missing  
  for( iMap1 =   scTrkAssocMap.begin(); iMap1 !=   scTrkAssocMap.end(); ++iMap1) {
    
    int nFound=0;
    for (  std::map<std::vector<reco::TransientTrack>, reco::CaloClusterPtr>::const_iterator iPair= allPairSCAss.begin(); iPair!= allPairSCAss.end(); ++iPair ) {
      if (  (iMap1->second) == (iPair->second)  ) nFound++;
    }
    
    if ( nFound == 0) {
      //      std::cout << " nFound zero case " << std::endl;
      int iList=0;
      for( iAux = auxMap.begin(); iAux!= auxMap.end(); ++iAux) {
	if ( (iMap1->first) == (iAux->second) && iList==0 )  {
	  thePair.clear();   
	  thePair.push_back(iAux->second);
	  allPairSCAss[thePair]= iMap1->second; 	
	  
	}
	
	iList++;
      }  
    }
    
  }




  // order the tracks in the pair in order of decreasing pt 
    for (  std::map<std::vector<reco::TransientTrack>,  reco::CaloClusterPtr>::const_iterator iPair= allPairSCAss.begin(); iPair!= allPairSCAss.end(); ++iPair ) {
      thePair.clear();
      if ( (iPair->first).size() ==2 ) {
	if (  sqrt((iPair->first)[0].track().innerMomentum().perp2()) > sqrt((iPair->first)[1].track().innerMomentum().perp2())  ) {
	  thePair.push_back((iPair->first)[0]);
	  thePair.push_back((iPair->first)[1]);
	} else {
	  thePair.push_back((iPair->first)[1]);
	  thePair.push_back((iPair->first)[0]);
	}
      } else {
	thePair.push_back((iPair->first)[0]);
      }

      allPairOrdInPtSCAss[thePair]=iPair->second;
    }


    //    std::cout << " ConversionTrackPairFinder FINAL allPairOrdInPtSCAss size " << allPairOrdInPtSCAss.size() << "\n";
    // for (  std::map<std::vector<reco::TransientTrack>,  reco::CaloClusterPtr>::const_iterator iPair= allPairOrdInPtSCAss.begin(); iPair!= allPairOrdInPtSCAss.end(); ++iPair ) {
    // std::cout << " ConversionTrackPairFindder FINAL allPairOrdInPtSCAss " << (iPair->first).size() << " SC Energy " << (iPair->second)->energy() << " eta " << (iPair->second)->eta() << " phi " <<  (iPair->second)->phi() << "\n";  
    // std::cout << " ConversionTrackPairFindder FINAL allPairOrdInPtSCAss (iPair->first).size() " << (iPair->first).size() << std::endl;
      
    // for ( std::vector<reco::TransientTrack>::const_iterator iTk=(iPair->first).begin(); iTk!= (iPair->first).end(); ++iTk) {
    //	std::cout << " ConversionTrackPair ordered track pt " << sqrt(iTk->track().innerMomentum().perp2()) << std::endl;
    // }
    //}

  

  return allPairOrdInPtSCAss;
 

}








