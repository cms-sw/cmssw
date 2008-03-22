 /*
 *  See header file for a description of this class.
 *  
 *
 *  $Date$
 *  $Revision$
 *
 *  \author: D. Pagano - University of Pavia & INFN Pavia
 */


#include "RecoMuon/MuonSeedGenerator/src/RPCSeedGenerator.h"
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedHits.h"
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedFinder.h"

// Data Formats 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

// Geometry
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "math.h"

// C++
#include <vector>

using namespace std;
using namespace edm;

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

double RB1X, RB1Y, RB1Z, RB2X, RB2Y, RB2Z, RB3X, RB3Y, RB3Z, RB4X, RB4Y, RB4Z;
double theta,L,s,r;

// Constructor
RPCSeedGenerator::RPCSeedGenerator(const edm::ParameterSet& pset){
  produces<TrajectorySeedCollection>(); 

  theRPCRecHits = pset.getParameter<edm::InputTag>("RPCRecHitsLabel");
  cout << endl << "[RPCSeedGenerator] --> Constructor called" << endl;
}  


// Destructor
RPCSeedGenerator::~RPCSeedGenerator(){
  cout << "[RPCSeedGenerator] --> Destructor called" << endl;
}


void RPCSeedGenerator::produce(edm::Event& event, const edm::EventSetup& eSetup){
  
  theSeeds.clear();

  // create the pointer to the Seed container
  auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  // Muon Geometry - DT, CSC and RPC 
  edm::ESHandle<MuonDetLayerGeometry> muonLayers;
  eSetup.get<MuonRecoGeometryRecord>().get(muonLayers);

  // get the RPC layers
  vector<DetLayer*> RPCBarrelLayers = muonLayers->barrelRPCLayers();
  const DetLayer* RB4L  = RPCBarrelLayers[5];
  const DetLayer* RB3L  = RPCBarrelLayers[4];
  const DetLayer* RB22L = RPCBarrelLayers[3];
  const DetLayer* RB21L = RPCBarrelLayers[2];
  const DetLayer* RB12L = RPCBarrelLayers[1];
  const DetLayer* RB11L = RPCBarrelLayers[0];


  MuonDetLayerMeasurements muonMeasurements(edm::InputTag(),edm::InputTag(), theRPCRecHits,
					    false, false, true);

  
  MuonRecHitContainer list11 = muonMeasurements.recHits(RB11L,event);
  MuonRecHitContainer list12 = muonMeasurements.recHits(RB12L,event);
  MuonRecHitContainer list21 = muonMeasurements.recHits(RB21L,event);
  MuonRecHitContainer list22 = muonMeasurements.recHits(RB22L,event);
  MuonRecHitContainer list3  = muonMeasurements.recHits(RB3L,event);
  MuonRecHitContainer list4  = muonMeasurements.recHits(RB4L,event); 


  if (list11.size() == 1 && list21.size() == 1 && list3.size() == 1 && list4.size() == 1) {

  
  cout << "list11 = " << list11.size() << endl;
  cout << "list21 = " << list21.size() << endl;
  cout << "list3 = " << list3.size() << endl;
  cout << "list4 = " << list4.size() << endl;
  

    unsigned int counter;
    
     bool* RB11 = 0;
       if (list11.size()) {
         RB11 = new bool[list11.size()];
         for ( size_t i=0; i<list11.size(); i++ ) RB11[i]=false;
       }

     bool* RB21 = 0;
       if (list21.size()) {
         RB21 = new bool[list21.size()];
         for ( size_t i=0; i<list21.size(); i++ ) RB21[i]=false;
       }


     bool* RB3 = 0;
       if (list3.size()) {
         RB3 = new bool[list3.size()];
         for ( size_t i=0; i<list3.size(); i++ ) RB3[i]=false;
       }


      for (MuonRecHitContainer::iterator iter=list4.begin(); iter!=list4.end(); iter++ ){
        RPCSeedFinder theSeed;
        theSeed.add(*iter);
        complete(theSeed, list3, RB3);
        complete(theSeed, list21, RB21);
        complete(theSeed, list11, RB11);
        checkAndFill(theSeed, eSetup);
      }
  
      for ( counter = 0; counter<list3.size(); counter++ ){
        if ( !RB3[counter] ) { 
          RPCSeedFinder theSeed;
          theSeed.add(list3[counter]);
          complete(theSeed, list21, RB21);
	  complete(theSeed, list11, RB11);
	  complete(theSeed, list4);
          checkAndFill(theSeed,eSetup);
        }
      }
  
      for ( counter = 0; counter<list21.size(); counter++ ){
        if ( !RB21[counter] ) { 
	  RPCSeedFinder theSeed;
	  theSeed.add(list21[counter]);
	  complete(theSeed, list11, RB11);
	  complete(theSeed, list4);
	  complete(theSeed, list3, RB3);
	    if (theSeed.nrhit()>1 || (theSeed.nrhit()==1 &&
	              		      theSeed.firstRecHit()->dimension()==4) ) {
	      checkAndFill(theSeed,eSetup);
             }
          }
        }
 
      for ( counter = 0; counter<list11.size(); counter++ ){
        if ( !RB11[counter] ) { 
  	  RPCSeedFinder theSeed;
	  theSeed.add(list11[counter]);
	  complete(theSeed, list4);
	  complete(theSeed, list3, RB3);
	  complete(theSeed, list21, RB21);
	    if (theSeed.nrhit()>1 || (theSeed.nrhit()==1 &&
	     			      theSeed.firstRecHit()->dimension()==4) ) {
	      checkAndFill(theSeed,eSetup);
	    }
          }
        }
  
    if ( RB3 ) delete [] RB3;
    if ( RB21 ) delete [] RB21;
    if ( RB11 ) delete [] RB11;

    if(theSeeds.size() == 1) output->push_back(theSeeds.front());
  
    else{
      for(vector<TrajectorySeed>::iterator seed = theSeeds.begin();
  	  seed != theSeeds.end(); ++seed){
        int counter =0;
        for(vector<TrajectorySeed>::iterator seed2 = seed;
	    seed2 != theSeeds.end(); ++seed2) 
	  if( seed->startingState().parameters().vector() ==
	      seed2->startingState().parameters().vector() )
	    ++counter;
      
        if( counter > 1 ) theSeeds.erase(seed--);
        else output->push_back(*seed);
      }
    }
    
    event.put(output);
    
  }

}


void RPCSeedGenerator::complete(RPCSeedFinder& seed,
                                MuonRecHitContainer &recHits, bool* used) const {

  MuonRecHitContainer good_rhit;
  
  ConstMuonRecHitPointer first = seed.firstRecHit(); 
  
  GlobalPoint ptg2 = first->globalPosition();
  
  int nr=0; // count rechits we have checked against seed
  
  for (MuonRecHitContainer::iterator iter=recHits.begin(); iter!=recHits.end(); iter++){
    
    GlobalPoint ptg1 = (*iter)->globalPosition();  //+v global pos of rechit
    
    if ( fabs (ptg1.eta()-ptg2.eta()) > .2 || fabs (ptg1.phi()-ptg2.phi()) > .1 ) {
      nr++;
      continue;
    }   // +vvp!!!
    
    if( fabs ( ptg2.eta() ) < 1.0 ) {    
     
      good_rhit.push_back(*iter);
	
    } 
    
    nr++;
    
  } 
  
  MuonRecHitPointer best=0;
  
  if( fabs ( ptg2.eta() ) < 1.0 ) {     
    
    float best_dphi = M_PI;
    
    for (MuonRecHitContainer::iterator iter=good_rhit.begin(); iter!=good_rhit.end(); iter++){
      
      GlobalVector dir1 = (*iter)->globalDirection();
      
      GlobalVector dir2 = first->globalDirection();
      
      float dphi = dir1.phi()-dir2.phi();
      
      if (dphi < 0.) dphi = -dphi;
      if (dphi > M_PI) dphi = 2.*M_PI - dphi;
      
      if (  dphi < best_dphi ) {
	
	best_dphi = dphi;
	best = (*iter);
      }
      
    }   
    
  }
  
  
  if(best)
    if ( best->isValid() ) seed.add(best);
  
} 


void RPCSeedGenerator::checkAndFill(RPCSeedFinder& theSeed, const edm::EventSetup& eSetup){
  
  if (theSeed.nrhit()>1 ) {
    vector<TrajectorySeed> the_seeds =  theSeed.seeds(eSetup);
    for (vector<TrajectorySeed>::const_iterator
	   the_seed=the_seeds.begin(); the_seed!=the_seeds.end(); ++the_seed) {
      theSeeds.push_back(*the_seed);
    }
  }
}
