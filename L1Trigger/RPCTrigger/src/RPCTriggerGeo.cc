// Class describing RPC trigger geometry
// aim: easly convert RPCdetId.firedStrip to loghit/logcone
// Author: Tomasz Fruboes


#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
//#include "Geometry/RPCGeometry/interface/RPCRollService.h" // Droped out in pre2
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "L1Trigger/RPCTrigger/src/RPCTriggerGeo.h"
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include "DataFormats/MuonDetId/interface/RPCDetId.h"


#include <vector>
#include <algorithm>
#include <map>

#include <cmath>

//#############################################################################
//
// Default constructor
//
//#############################################################################
RPCTriggerGeo::RPCTriggerGeo(){ 
  isGeoBuild=false;
}

//#############################################################################
//
// Checks, if we have build the geometry already
//
//#############################################################################
bool RPCTriggerGeo::isGeometryBuild(){
  return isGeoBuild;
}

//#############################################################################
//
// Builds RpcGeometry
// code accessing geometry info is heavly based 
// on CMSSW/Geometry/RPCGeometry/test/RPCGeometryAnalyzer.cc
//
//#############################################################################
void RPCTriggerGeo::buildGeometry(edm::ESHandle<RPCGeometry> rpcGeom){
 
  
  std::cout << "Building RPC geometry" << std::endl;  // Check how to give
                                                      // output in a kosher way
  // Get some information for all RpcDetId`s; store it locally
  TrackingGeometry::DetContainer::const_iterator it;
  for(it = rpcGeom->dets().begin();
      it != rpcGeom->dets().end();
      it++)
  {
    
    if( dynamic_cast< RPCRoll* >( *it ) == 0 ) continue;
    RPCRoll* roll = dynamic_cast< RPCRoll*>( *it );
    addDet(roll);
    
  } // RpcDet loop end

  // Build RpcCurl's




  isGeoBuild=true;
  printCurlMapInfo();
  
}

//#############################################################################
//
// Adds detID to the collection
//
//#############################################################################
void RPCTriggerGeo::addDet(RPCRoll* roll){

  RPCDetId detId = roll->id();
  
  std::vector<Local3DPoint> edges;
  //std::vector<GlobalPoint> edges;
  std::vector<float> etas;  

  const StripTopology* topology = dynamic_cast<const StripTopology*>
                                  (&(roll->topology()));
  
  float stripLength = topology->localStripLength(LocalPoint( 0., 0., 0. ));    

  // The list of chamber local positions used to find etaMin and etaMax
  // of a chamber. You can add as many points as desire, but make sure
  // the point lays _inside_ the chamber.
  // FIXME: Current method doesnt work
  

  //*  // using y as nonzero doesnt help
  edges.push_back(Local3DPoint( 0., 0., stripLength/2.)); // Add (?) correction for
  edges.push_back(Local3DPoint( 0., 0.,-stripLength/2.)); // nonzero chamber height

  for (unsigned int i=0; i < edges.size(); i++){
    GlobalPoint gp = roll->toGlobal( edges[i] );
    etas.push_back( gp.eta() );
  }
  //*/

  /* // Possible soltuion to eta problem - use global point
    // Doesnt work :(
  LocalPoint lpCentre(0., 0., 0.);
  GlobalPoint gpCentre = roll->toGlobal( lpCentre );

  float x = gpCentre.x();
  float y = gpCentre.y();
  float z = gpCentre.z();

  if (detId.region()==0){ // Barell
    edges.push_back( GlobalPoint(x, y, z + stripLength/2.) );
    edges.push_back( GlobalPoint(x, y, z - stripLength/2.) );

  }

  for (unsigned int i=0; i < edges.size(); i++){
    GlobalPoint gp = edges[i];
    etas.push_back( gp.eta() );
  }
  //*/
  


  //RPCDetInfo(uint32_t mDetId, int region, int ring, int station, int layer, int roll);
  RPCDetInfo detInfo(detId.rawId(), detId.region(), detId.ring(), 
                     detId.station(),  detId.layer(), detId.roll() );
  
  
  detInfo.mEtaMin = *( min_element(etas.begin(), etas.end()) );
  detInfo.mEtaMax = *( max_element(etas.begin(), etas.end()) );
  
  
  
  
  if( mRPCCurlMap.find(detInfo.getCurlId() ) != mRPCCurlMap.end() ){ // Curl allready in map
     mRPCCurlMap[detInfo.getCurlId()].addDetId(detInfo);

  } else {  // new curl
    
    RPCCurl newCurl;
    newCurl.addDetId(detInfo);
    mRPCCurlMap[detInfo.getCurlId()]=newCurl;

  }



}
//#############################################################################
//
// Util function to print rpcChambersMap contents
//
//#############################################################################
void RPCTriggerGeo::printCurlMapInfo(){ // XXX - Erase ME
  
  RPCCurlMap::const_iterator it;
  for ( it=mRPCCurlMap.begin(); it != mRPCCurlMap.end(); it++){
    std::cout << "------------------------------"<< std::endl;
    std::cout << "CurlId " << (it->first) << " " << std::endl;
    (it->second).printContents();
                 //printContents
  }


}

//#############################################################################
//
//  
//
//#############################################################################



