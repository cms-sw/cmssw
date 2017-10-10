/* This is CSCHitFromWireOnly
 *
 * Finds wiregroup with hits, and fill in CSCWireHitCollection
 * which includes only DetId and wiregroup #
 *
 */

#include <RecoLocalMuon/CSCRecHitD/src/CSCHitFromWireOnly.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHit.h>

#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>

#include <DataFormats/CSCDigi/interface/CSCWireDigi.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>


CSCHitFromWireOnly::CSCHitFromWireOnly( const edm::ParameterSet& ps ) : recoConditions_(nullptr){
  
  deltaT                 = ps.getParameter<int>("CSCWireClusterDeltaT");
  useReducedWireTime     = ps.getParameter<bool>("CSCUseReducedWireTimeWindow");
  wireTimeWindow_low     = ps.getParameter<int>("CSCWireTimeWindowLow");
  wireTimeWindow_high     = ps.getParameter<int>("CSCWireTimeWindowHigh");

  //clusterSize            = ps.getParameter<int>("CSCWireClusterMaxSize");
}


CSCHitFromWireOnly::~CSCHitFromWireOnly(){}


std::vector<CSCWireHit> CSCHitFromWireOnly::runWire( const CSCDetId& id, const CSCLayer* layer, const CSCWireDigiCollection::Range& rwired ) {
  
  std::vector<CSCWireHit> hitsInLayer;

  id_        = id;
  layer_ = layer;
  layergeom_ = layer->geometry();
  bool any_digis = true;
  int n_wgroup = 0;


  // Loop over wire digi collection
  for ( CSCWireDigiCollection::const_iterator it = rwired.first; it != rwired.second; ++it ) {
    
    const CSCWireDigi wdigi = *it;

    if(isDeadWG( id, wdigi.getWireGroup())){ 	 
      continue; 	 
    }
    if ( any_digis ) {
      any_digis = false;
      makeWireCluster( wdigi );
      n_wgroup = 1;
    } else {
      if ( !addToCluster( wdigi ) ) {
	      // Make Wire Hit from cluster, delete old cluster and start new one
	float whit_pos = findWireHitPosition();
      	bool deadWG_left = isDeadWG( id, wire_in_cluster.at(0) -1 ); 
	bool deadWG_right = isDeadWG( id, wire_in_cluster.at(wire_in_cluster.size()-1) + 1);
	short int aDeadWG = 0;
	if(!deadWG_left && !deadWG_right){
	  aDeadWG = 0;
	}
	else if(deadWG_left && deadWG_right){
	  aDeadWG = 255;
	}
	else{
	  if(deadWG_left){
	    aDeadWG = wire_in_cluster.at(0) -1;
	  }
	  else{
	    aDeadWG = wire_in_cluster.at(wire_in_cluster.size()-1) + 1;
	  }
	}
       // Set time bins for wire hit as the time bins of the central wire digi, lower of central two if an even number of digis.
      std::vector <int> timeBinsOn=wire_cluster[n_wgroup/2].getTimeBinsOn();
      //CSCWireHit whit(id, whit_pos, wire_in_clusterAndBX, theTime, isDeadWGAround, timeBinsOn );
      CSCWireHit whit(id, whit_pos, wire_in_clusterAndBX, theTime, aDeadWG, timeBinsOn );

      if (!useReducedWireTime) {
        hitsInLayer.push_back( whit );
      }
      else if (theTime >= wireTimeWindow_low && theTime <= wireTimeWindow_high) {
        hitsInLayer.push_back( whit );	
      }

      makeWireCluster( wdigi );
      n_wgroup = 1;
      } else {
	      n_wgroup++;
      }
    }
    // Don't forget to fill last wire hit !!!
    if ( rwired.second - it == 1) {           
      float whit_pos = findWireHitPosition();
      bool deadWG_left = isDeadWG( id, wire_in_cluster.at(0) -1 ); 
      bool deadWG_right = isDeadWG( id, wire_in_cluster.at(wire_in_cluster.size()-1) + 1); 
      short int aDeadWG = 0;
      if(!deadWG_left && !deadWG_right){
        aDeadWG = 0;
      }
      else if(deadWG_left && deadWG_right){
        aDeadWG = 255;
      }
      else{
        if(deadWG_left){
          aDeadWG = wire_in_cluster.at(0) -1;
        }
        else{
          aDeadWG = wire_in_cluster.at(wire_in_cluster.size()-1) + 1;
        }
      }
      std::vector <int> timeBinsOn=wire_cluster[n_wgroup/2].getTimeBinsOn();
      /// BX
      //CSCWireHit whit(id, whit_pos, wire_in_clusterAndBX, theTime, isDeadWGAround, timeBinsOn );
      CSCWireHit whit(id, whit_pos, wire_in_clusterAndBX, theTime, aDeadWG, timeBinsOn );

      if (!useReducedWireTime) {
        hitsInLayer.push_back( whit );
      } 
      else if (theTime >= wireTimeWindow_low && theTime <= wireTimeWindow_high) {
        hitsInLayer.push_back( whit );
      }

      n_wgroup++;
    }
  }

/// Print statement (!!!to control WireHit content!!!) BX
  /*
      for(std::vector<CSCWireHit>::const_iterator itWHit=hitsInLayer.begin(); itWHit!=hitsInLayer.end(); ++itWHit){
         (*itWHit).print(); 
         }  
  */

  return hitsInLayer;
}


void CSCHitFromWireOnly::makeWireCluster(const CSCWireDigi & digi) {
  wire_cluster.clear();
  wire_in_cluster.clear();
  wire_in_clusterAndBX.clear(); /// BX to wire
  theLastChannel  = digi.getWireGroup();
  theTime         = digi.getTimeBin();
  wire_cluster.push_back( digi );
}


bool CSCHitFromWireOnly::addToCluster(const CSCWireDigi & digi) {

  
  int iwg = digi.getWireGroup();
  
  if ( iwg == theLastChannel ){
    return true;  // Same wire group but different tbin -> ignore
  }
  else{
    if ( (iwg == theLastChannel+1) && (abs(digi.getTimeBin()-theTime)<= deltaT) ) {
      theLastChannel = iwg;
      wire_cluster.push_back( digi );
      return true;
    }
  }
  
  return false;
}


/* findWireHitPosition
 *
 * This position is expressed in terms of wire #... is a float since it may be a fraction.
 */
float CSCHitFromWireOnly::findWireHitPosition() {
  
  // Again use center of mass to determine position of wire hit
  // To do so, need to know wire spacing and # of wires
  
  float y = 0.0;
  
  for ( unsigned i = 0; i < wire_cluster.size(); ++i ) {
    CSCWireDigi wdigi = wire_cluster[i];
    int wgroup = wdigi.getWireGroup();
    wire_in_cluster.push_back( wgroup );
    int wgroupAndBX = wdigi.getBXandWireGroup(); /// BX to WireHit
    //std::cout << " wgroupAndBX: " << std::hex << wgroupAndBX << std::dec << std::endl;
    wire_in_clusterAndBX.push_back( wgroupAndBX ); /// BX to WireHit
    y += float( wgroup );
  }       

  float wiregpos = y /wire_cluster.size() ;

  return wiregpos;

}

bool CSCHitFromWireOnly::isDeadWG(const CSCDetId& id, int WG){

  const std::bitset<112> & deadWG = recoConditions_->badWireWord( id );
  bool isDead = false;
  if(WG>-1 && WG<112){
    isDead = deadWG.test(WG);
  }
  return isDead;
} 
