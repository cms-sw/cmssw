/** \file RPCTriggerGeo.cc
 *
 *  $Date: 2006/05/29 12:00:00 $
 *  $Revision: 1.1 $
 *  \author Tomasz Fruboes
 */

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
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
/**
 *
 * \brief Default constructor
 *
*/
//#############################################################################
RPCTriggerGeo::RPCTriggerGeo(){

  m_isGeometryBuilt=false;

}
//#############################################################################
/**
*
* \brief Checks, if we have built the geometry already
*
*/
//#############################################################################
bool RPCTriggerGeo::isGeometryBuilt(){

  return m_isGeometryBuilt;

}

//#############################################################################
/**
 * \brief Builds RpcGeometry
 * \note Code accessing geometry info is heavly based on
           CMSSW/Geometry/RPCGeometry/test/RPCGeometryAnalyzer.cc
 *
*/
//#############################################################################
void RPCTriggerGeo::buildGeometry(edm::ESHandle<RPCGeometry> rpcGeom){
 
  
  //std::cout << "Building RPC geometry" << std::endl;  // Check how to give
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


  m_isGeometryBuilt=true;
  printCurlMapInfo();
}

//#############################################################################
/**
 *
 * \brief Adds detID to the collection
 * \bug Method used to calculate minTower and maxTower is broken
 *
*/
//#############################################################################
void RPCTriggerGeo::addDet(RPCRoll* roll){

  RPCDetInfo detInfo(roll);

  if( m_RPCCurlMap.find(detInfo.getCurlId()) != m_RPCCurlMap.end() ){ // Curl allready in map

     m_RPCCurlMap[detInfo.getCurlId()].addDetId(detInfo);

  } else {  // add a new curl
    
    RPCCurl newCurl;
    newCurl.addDetId(detInfo);
    m_RPCCurlMap[detInfo.getCurlId()]=newCurl;

  }

}
//#############################################################################
/**
*
* \brief Util function to print rpcChambersMap contents
* \note Since cout`s are forbidden most of code is commented out
*
*/
//#############################################################################
void RPCTriggerGeo::printCurlMapInfo(){ // XXX - Erase ME
  
  RPCCurlMap::const_iterator it;
  for ( it=m_RPCCurlMap.begin(); it != m_RPCCurlMap.end(); it++){
  /*
    std::cout << "------------------------------"<< std::endl;
    std::cout << "CurlId " << (it->first) << " " << std::endl;
    (it->second).printContents();
                 //printContents
  //*/
  }


}
