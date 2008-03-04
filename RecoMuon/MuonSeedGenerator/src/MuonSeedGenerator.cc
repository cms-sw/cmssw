/**
 *  See header file for a description of this class.
 *  
 *  All the code is under revision
 *
 *  $Date$
 *  $Revision$
 *
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author ported by: R. Bellan - INFN Torino
 */


#include "RecoMuon/MuonSeedGenerator/src/MuonSeedGenerator.h"

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFinder.h"

// Data Formats 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

#include "DataFormats/Common/interface/Handle.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

// C++
#include <vector>

using namespace std;

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

// Constructor
MuonSeedGenerator::MuonSeedGenerator(const edm::ParameterSet& pset){
  produces<TrajectorySeedCollection>(); 

  // enable the DT chamber
  enableDTMeasurement = pset.getParameter<bool>("EnableDTMeasurement");

  // enable the CSC chamber
  enableCSCMeasurement = pset.getParameter<bool>("EnableCSCMeasurement");

  if(enableDTMeasurement)
    // the name of the DT rec hits collection
    theDTRecSegmentLabel = pset.getParameter<edm::InputTag>("DTRecSegmentLabel");

  if(enableCSCMeasurement)
    // the name of the CSC rec hits collection
    theCSCRecSegmentLabel = pset.getParameter<edm::InputTag>("CSCRecSegmentLabel");
}

// Destructor
MuonSeedGenerator::~MuonSeedGenerator(){}


// reconstruct muon's seeds
void MuonSeedGenerator::produce(edm::Event& event, const edm::EventSetup& eSetup){
  theSeeds.clear();
  
  // create the pointer to the Seed container
  auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  // divide the RecHits by DetLayer, in order to fill the
  // RecHitContainer like it was in ORCA
  
  // Muon Geometry - DT, CSC and RPC 
  edm::ESHandle<MuonDetLayerGeometry> muonLayers;
  eSetup.get<MuonRecoGeometryRecord>().get(muonLayers);

  // get the DT layers
  vector<DetLayer*> dtLayers = muonLayers->allDTLayers();

  // get the CSC layers
  vector<DetLayer*> cscForwardLayers = muonLayers->forwardCSCLayers();
  vector<DetLayer*> cscBackwardLayers = muonLayers->backwardCSCLayers();
    
  // Backward (z<0) EndCap disk
  const DetLayer* ME4Bwd = cscBackwardLayers[4];
  const DetLayer* ME3Bwd = cscBackwardLayers[3];
  const DetLayer* ME2Bwd = cscBackwardLayers[2];
  const DetLayer* ME12Bwd = cscBackwardLayers[1];
  const DetLayer* ME11Bwd = cscBackwardLayers[0];
  
  // Forward (z>0) EndCap disk
  const DetLayer* ME11Fwd = cscForwardLayers[0];
  const DetLayer* ME12Fwd = cscForwardLayers[1];
  const DetLayer* ME2Fwd = cscForwardLayers[2];
  const DetLayer* ME3Fwd = cscForwardLayers[3];
  const DetLayer* ME4Fwd = cscForwardLayers[4];
     
  // barrel
  const DetLayer* MB4DL = dtLayers[3];
  const DetLayer* MB3DL = dtLayers[2];
  const DetLayer* MB2DL = dtLayers[1];
  const DetLayer* MB1DL = dtLayers[0];
  
  // instantiate the accessor
  // Don not use RPC for seeding
  MuonDetLayerMeasurements muonMeasurements(theDTRecSegmentLabel.label(),theCSCRecSegmentLabel,edm::InputTag(),
					    enableDTMeasurement,enableCSCMeasurement,false);

  // ------------        EndCap disk z<0 + barrel

  MuonRecHitContainer list24 = muonMeasurements.recHits(ME4Bwd,event);
  MuonRecHitContainer list23 = muonMeasurements.recHits(ME3Bwd,event);
  
  MuonRecHitContainer list12 = muonMeasurements.recHits(ME2Bwd,event);
  
  MuonRecHitContainer list22 = muonMeasurements.recHits(ME12Bwd,event);
  MuonRecHitContainer list21 = muonMeasurements.recHits(ME11Bwd,event);

  MuonRecHitContainer list11 = list21; 
  MuonRecHitContainer list5 = list22;
  MuonRecHitContainer list13 = list23;  
  MuonRecHitContainer list4 = list24; 
 
  if ( list21.size() == 0 )  { 
    list11 = list22; list5 = list21;
  }

  if ( list24.size() < list23.size() && list24.size() > 0 )  { 
    list13 = list24; list4 = list23;
  }

  if ( list23.size() == 0 )  { 
    list13 = list24; list4 = list23;
  }

  MuonRecHitContainer list1 = list11;
  MuonRecHitContainer list2 = list12;
  MuonRecHitContainer list3 = list13;


  if ( list12.size() == 0 )  { 
    list3 = list12;
    if ( list11.size() <= list13.size() && list11.size() > 0 ) {
      list1 = list11; list2 = list13;}
    else { list1 = list13; list2 = list11;}
  }

  if ( list13.size() == 0 )  { 
    if ( list11.size() <= list12.size() && list11.size() > 0 ) {
      list1 = list11; list2 = list12;}
    else { list1 = list12; list2 = list11;}
  }
   
  if ( list12.size() != 0 &&  list13.size() != 0 )  { 
    if ( list11.size()<=list12.size() && list11.size()<=list13.size() && list11.size()>0 ) {   // ME 1
      if ( list12.size() > list13.size() ) {
	list2 = list13; list3 = list12;}
    }
    else if ( list12.size() <= list13.size() ) {                                   //  start with ME 2
      list1 = list12;
      if ( list11.size() <= list13.size() && list11.size() > 0 ) {
	list2 = list11; list3 = list13;}
      else { list2 = list13; list3 = list11;}
    } 
    else {                                                                         //  start with ME 3
      list1 = list13;
      if ( list11.size() <= list12.size() && list11.size() > 0 ) {
	list2 = list11; list3 = list12;}
      else { list2 = list12; list3 = list11;}
    }
  }

  MuonRecHitContainer list6 = muonMeasurements.recHits(MB3DL,event);
  MuonRecHitContainer list7 = muonMeasurements.recHits(MB2DL,event);
  MuonRecHitContainer list8 = muonMeasurements.recHits(MB1DL,event);
  
  bool* MB1 = zero(list8.size());
  bool* MB2 = zero(list7.size());
  bool* MB3 = zero(list6.size());

  bool* ME2 = zero(list2.size());
  bool* ME3 = zero(list3.size());
  bool* ME4 = zero(list4.size());
  bool* ME5 = zero(list5.size());

  // creates list of compatible track segments

  for (MuonRecHitContainer::iterator iter = list1.begin(); iter!=list1.end(); iter++ ){
    if ( (*iter)->recHits().size() < 4 && list3.size() > 0 ) continue; // 3p.tr-seg. are not so good for starting
    MuonSeedFinder theSeed;
    theSeed.add(*iter);
    complete(theSeed, list2, ME2);
    complete(theSeed, list3, ME3);
    complete(theSeed, list4, ME4);
    complete(theSeed, list5, ME5);
    complete(theSeed, list6, MB3);
    complete(theSeed, list7, MB2);    
    complete(theSeed, list8, MB1);
    checkAndFill(theSeed,eSetup);
  }


  unsigned int counter;

  for ( counter = 0; counter<list2.size(); counter++ ){

    if ( !ME2[counter] ) {
      MuonSeedFinder theSeed;
      theSeed.add(list2[counter]);
      complete(theSeed, list3, ME3);
      complete(theSeed, list4, ME4);
      complete(theSeed, list5, ME5);
      complete(theSeed, list6, MB3);
      complete(theSeed, list7, MB2);
      complete(theSeed, list8, MB1);

      checkAndFill(theSeed,eSetup);
    }
  }


  if ( list3.size() < 20 ) {   // +v
    for ( counter = 0; counter<list3.size(); counter++ ){
      if ( !ME3[counter] ) { 
	MuonSeedFinder theSeed;
	theSeed.add(list3[counter]);
	complete(theSeed, list4, ME4);
	complete(theSeed, list5, ME5);
	complete(theSeed, list6, MB3);
	complete(theSeed, list7, MB2);
	complete(theSeed, list8, MB1);
	
	checkAndFill(theSeed,eSetup);
      }
    }
  }

  if ( list4.size() < 20 ) {   // +v
    for ( counter = 0; counter<list4.size(); counter++ ){
      if ( !ME4[counter] ) {
	MuonSeedFinder theSeed;
	theSeed.add(list4[counter]);
	complete(theSeed, list5, ME5);
	complete(theSeed, list6, MB3);
	complete(theSeed, list7, MB2);
	complete(theSeed, list8, MB1);

	checkAndFill(theSeed,eSetup);
      }   
    }          
  } 

  // ------------        EndCap disk z>0

  list24 = muonMeasurements.recHits(ME4Fwd,event);
  list23 = muonMeasurements.recHits(ME3Fwd,event);
  
  list12 = muonMeasurements.recHits(ME2Fwd,event);
  
  list22 = muonMeasurements.recHits(ME12Fwd,event);
  list21 = muonMeasurements.recHits(ME11Fwd,event);
  
 
  list11 = list21; 
  list5 = list22;
  list13 = list23;  
  list4 = list24; 

  if ( list21.size() == 0 )  { 
    list11 = list22; list5 = list21;
  }

  if ( list24.size() < list23.size() && list24.size() > 0 )  { 
    list13 = list24; list4 = list23;
  }

  if ( list23.size() == 0 )  { 
    list13 = list24; list4 = list23;
  }

  list1 = list11;
  list2 = list12;
  list3 = list13;


  if ( list12.size() == 0 )  { 
    list3 = list12;
    if ( list11.size() <= list13.size() && list11.size() > 0 ) {
      list1 = list11; list2 = list13;}
    else { list1 = list13; list2 = list11;}
  }

  if ( list13.size() == 0 )  { 
    if ( list11.size() <= list12.size() && list11.size() > 0 ) {
      list1 = list11; list2 = list12;}
    else { list1 = list12; list2 = list11;}
  }
   
  if ( list12.size() != 0 &&  list13.size() != 0 )  { 
    if ( list11.size()<=list12.size() && list11.size()<=list13.size() && list11.size()>0 ) {  // ME 1
      if ( list12.size() > list13.size() ) {
	list2 = list13; list3 = list12;}
    }
    else if ( list12.size() <= list13.size() ) {                                  //  start with ME 2
      list1 = list12;
      if ( list11.size() <= list13.size() && list11.size() > 0 ) {
	list2 = list11; list3 = list13;}
      else { list2 = list13; list3 = list11;}
    } 
    else {                                                                        //  start with ME 3
      list1 = list13;
      if ( list11.size() <= list12.size() && list11.size() > 0 ) {
	list2 = list11; list3 = list12;}
      else { list2 = list12; list3 = list11;}
    }
  }


  if ( ME5 ) delete [] ME5;
  if ( ME4 ) delete [] ME4;
  if ( ME3 ) delete [] ME3;
  if ( ME2 ) delete [] ME2;

  ME2 = zero(list2.size());
  ME3 = zero(list3.size());
  ME4 = zero(list4.size());
  ME5 = zero(list5.size());


  for (MuonRecHitContainer::iterator iter=list1.begin(); iter!=list1.end(); iter++ ){
    if ( (*iter)->recHits().size() < 4 && list3.size() > 0 ) continue;// 3p.tr-seg.aren't so good for starting
    MuonSeedFinder theSeed;
    theSeed.add(*iter);
    complete(theSeed, list2, ME2);
    complete(theSeed, list3, ME3);
    complete(theSeed, list4, ME4);
    complete(theSeed, list5, ME5);
    complete(theSeed, list6, MB3);
    complete(theSeed, list7, MB2);
    complete(theSeed, list8, MB1);

    checkAndFill(theSeed,eSetup);
    
  }


  for ( counter = 0; counter<list2.size(); counter++ ){
    if ( !ME2[counter] ) {
      MuonSeedFinder theSeed;
      theSeed.add(list2[counter]);
      complete(theSeed, list3, ME3);
      complete(theSeed, list4, ME4);
      complete(theSeed, list5, ME5);
      complete(theSeed, list6, MB3);
      complete(theSeed, list7, MB2);
      complete(theSeed, list8, MB1);

      checkAndFill(theSeed,eSetup);
    } 
  }


  if ( list3.size() < 20 ) {   // +v
    for ( counter = 0; counter<list3.size(); counter++ ){
      if ( !ME3[counter] ) { 
	MuonSeedFinder theSeed;
	theSeed.add(list3[counter]);
	complete(theSeed, list4, ME4);
	complete(theSeed, list5, ME5);
	complete(theSeed, list6, MB3);
	complete(theSeed, list7, MB2);
	complete(theSeed, list8, MB1);

	checkAndFill(theSeed,eSetup);
      }
    }
  }

  if ( list4.size() < 20 ) {   // +v
    for ( counter = 0; counter<list4.size(); counter++ ){
      if ( !ME4[counter] ) {
	MuonSeedFinder theSeed;
	theSeed.add(list4[counter]);
	complete(theSeed, list5, ME5);
	complete(theSeed, list6, MB3);
	complete(theSeed, list7, MB2);
	complete(theSeed, list8, MB1);

	checkAndFill(theSeed,eSetup);
      }   
    }          
  } 


  // ----------    Barrel only
  
  MuonRecHitContainer list9 = muonMeasurements.recHits(MB4DL,event);

  if ( list9.size() < 100 ) {   // +v
    for (MuonRecHitContainer::iterator iter=list9.begin(); iter!=list9.end(); iter++ ){
      MuonSeedFinder theSeed;
      theSeed.add(*iter);
      complete(theSeed, list6, MB3);
      complete(theSeed, list7, MB2);
      complete(theSeed, list8, MB1);

      checkAndFill(theSeed,eSetup);
    }
  }


  if ( list6.size() < 100 ) {   // +v
    for ( counter = 0; counter<list6.size(); counter++ ){
      if ( !MB3[counter] ) { 
	MuonSeedFinder theSeed;
	theSeed.add(list6[counter]);
	complete(theSeed, list7, MB2);
	complete(theSeed, list8, MB1);
	complete(theSeed, list9);

	checkAndFill(theSeed,eSetup);
      }
    }
  }


  if ( list7.size() < 100 ) {   // +v
    for ( counter = 0; counter<list7.size(); counter++ ){
      if ( !MB2[counter] ) { 
	MuonSeedFinder theSeed;
	theSeed.add(list7[counter]);
	complete(theSeed, list8, MB1);
	complete(theSeed, list9);
	complete(theSeed, list6, MB3);
	if (theSeed.nrhit()>1 || (theSeed.nrhit()==1 &&
				  theSeed.firstRecHit()->dimension()==4) ) {
	  fill(theSeed,eSetup);
	}
      }
    }
  }


  if ( list8.size() < 100 ) {   // +v
    for ( counter = 0; counter<list8.size(); counter++ ){
      if ( !MB1[counter] ) { 
	MuonSeedFinder theSeed;
	theSeed.add(list8[counter]);
	complete(theSeed, list9);
	complete(theSeed, list6, MB3);
	complete(theSeed, list7, MB2);
	if (theSeed.nrhit()>1 || (theSeed.nrhit()==1 &&
				  theSeed.firstRecHit()->dimension()==4) ) {
	  fill(theSeed,eSetup);
	}
      }
    }
  }

  if ( ME5 ) delete [] ME5;
  if ( ME4 ) delete [] ME4;
  if ( ME3 ) delete [] ME3;
  if ( ME2 ) delete [] ME2;
  if ( MB3 ) delete [] MB3;
  if ( MB2 ) delete [] MB2;
  if ( MB1 ) delete [] MB1;

  
  if(theSeeds.size() == 1)
    output->push_back(theSeeds.front());
  
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


bool * MuonSeedGenerator::zero(unsigned listSize)
{
  bool * result = 0;
  if (listSize) {
    result = new bool[listSize]; 
    for ( size_t i=0; i<listSize; i++ ) result[i]=false;
  }
  return result;
}


void MuonSeedGenerator::complete(MuonSeedFinder& seed,
                                 MuonRecHitContainer &recHits, bool* used) const {

  MuonRecHitContainer good_rhit;

  //+v get all rhits compatible with the seed on dEta/dPhi Glob.

  ConstMuonRecHitPointer first = seed.firstRecHit(); // first rechit of seed

  GlobalPoint ptg2 = first->globalPosition(); // its global pos +v

  int nr=0; // count rechits we have checked against seed

  for (MuonRecHitContainer::iterator iter=recHits.begin(); iter!=recHits.end(); iter++){

    GlobalPoint ptg1 = (*iter)->globalPosition();  //+v global pos of rechit
    float deta = fabs (ptg1.eta()-ptg2.eta());
    // Geom::Phi should keep it in the range [-pi, pi]
    float dphi = fabs (ptg1.phi()-ptg2.phi());
    float eta2 = fabs( ptg2.eta() );

    // Cox: Just too far away?
    if ( deta > .2 || dphi > .1 ) {
      nr++;
      continue;
    }   // +vvp!!!

    if( eta2 < 1.0 ) {     //  barrel only

      LocalPoint pt1 = first->det()->toLocal(ptg1); // local pos of rechit in seed's det

      LocalVector dir1 = first->localDirection();

      LocalPoint pt2 = first->localPosition();

      float m = dir1.z()/dir1.x();   // seed's slope in local xz
      float yf = pt1.z();            // local z of rechit
      float yi = pt2.z();            // local z of seed
      float xi = pt2.x();            // local x of seed
      float xf = (yf-yi)/m + xi;     // x of linear extrap alone seed direction to z of rechit
      float dist = fabs ( xf - pt1.x() ); // how close is actual to predicted local x ?

      float d_cut = sqrt((yf-yi)*(yf-yi)+(pt1.x()-pt2.x())*(pt1.x()-pt2.x()))/10.;


      //@@ Tim asks: what is the motivation for this cut?
      //@@ It requires (xpred-xrechit)< 0.1 * distance between rechit and seed in xz plane
      if ( dist < d_cut ) {
	good_rhit.push_back(*iter);
	if (used) used[nr]=true;
      }

    }  // eta  < 1.0

    else {    //  endcap & overlap.
      // allow a looser dphi cut where bend is greatest, so we get those little 5-GeV muons
      // watch out for ghosts from ME1/A, below 2.0.
      float dphicut = (eta2 > 1.6 && eta2 < 2.0) ? 0.1 : 0.07;
      // segments at the edge of the barrel may not have a good eta measurement
      float detacut = (first->isDT() || (*iter)->isDT()) ? 0.2 : 0.1;

      if ( deta < detacut && dphi < dphicut ) {
	good_rhit.push_back(*iter);
	if (used) used[nr]=true;
      }

    }  // eta > 1.0


    nr++;

  }  // recHits iter

  // select the best rhit among the compatible ones (based on Dphi Glob & Dir)

  MuonRecHitPointer best=0;

  float best_dphiG = M_PI;
  float best_dphiD = M_PI;

  if( fabs ( ptg2.eta() ) > 1.0 ) {    //  endcap & overlap.
      
    // select the best rhit among the compatible ones (based on Dphi Glob & Dir)
      
    GlobalVector dir2 =  first->globalDirection();
   
    GlobalPoint  pos2 =  first->globalPosition();  // +v
      
    for (MuonRecHitContainer::iterator iter=good_rhit.begin(); iter!=good_rhit.end(); iter++){

      GlobalPoint pos1 = (*iter)->globalPosition();  // +v
 
      float dphi = pos1.phi()-pos2.phi();       //+v

      if (dphi < 0.) dphi = -dphi;             //+v
      if (dphi > M_PI) dphi = 2.*M_PI - dphi;  //+v

      if (  dphi < best_dphiG*1.5 ) {  


	if (  dphi < best_dphiG*.67  && best_dphiG > .005 )  best_dphiD = M_PI;  // thresh. of strip order

	GlobalVector dir1 = (*iter)->globalDirection();
	
	float  dphidir = fabs ( dir1.phi()-dir2.phi() );

	if (dphidir > M_PI) dphidir = 2.*M_PI - dphidir;
	if (dphidir > M_PI*.5) dphidir = M_PI - dphidir;  // +v  [0,pi/2]
	if (  dphidir < best_dphiD ) {

	  best_dphiG = dphi;
	  if ( dphi < .002 )  best_dphiG =  .002;                          // thresh. of half-strip order
	  best_dphiD = dphidir;
	  best = (*iter);

	}

      }


    }   //  rhit iter

  }  // eta > 1.0

  if( fabs ( ptg2.eta() ) < 1.0 ) {     //  barrel only

    // select the best rhit among the compatible ones (based on Dphi)

    float best_dphi = M_PI;

    for (MuonRecHitContainer::iterator iter=good_rhit.begin(); iter!=good_rhit.end(); iter++){
      GlobalVector dir1 = (*iter)->globalDirection();

      //@@ Tim: Why do this again? 'first' hasn't changed, has it?
      //@@ I comment it out.
      //    RecHit first = seed.rhit();
      
      GlobalVector dir2 = first->globalDirection();
      
      float dphi = dir1.phi()-dir2.phi();

      if (dphi < 0.) dphi = -dphi;
      if (dphi > M_PI) dphi = 2.*M_PI - dphi;

      if (  dphi < best_dphi ) {

	best_dphi = dphi;
	best = (*iter);
      }

    }   //  rhit iter

  }  // eta < 1.0


  // add the best Rhit to the seed 
  if(best)
    if ( best->isValid() ) seed.add(best);

}  //   void complete.


void MuonSeedGenerator::checkAndFill(MuonSeedFinder& seedFinder, const edm::EventSetup& eSetup){

  if (seedFinder.nrhit()>1 ) {
    fill(seedFinder, eSetup);
  }
}


void MuonSeedGenerator::fill(MuonSeedFinder& seedFinder, const edm::EventSetup& eSetup)
{
  vector<TrajectorySeed> seeds  = seedFinder.seeds(eSetup);
  for (vector<TrajectorySeed>::const_iterator seedItr = seeds.begin(), seedEnd = seeds.end();
       seedItr != seedEnd; ++seedItr)
  {
    // FIXME, ask for this method
    //if ( (*the_seed).isValid() )
    theSeeds.push_back(*seedItr);
  }
}

