/** \file RPCTriggerGeo.cc
 *
 *  $Date: 2006/06/06 16:25:00 $
 *  $Revision: 1.5 $
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
 
  // Get some information for all RpcDetId`s; store it locally
  for(TrackingGeometry::DetContainer::const_iterator it = rpcGeom->dets().begin();
      it != rpcGeom->dets().end();
      it++)
  {
    
    if( dynamic_cast< RPCRoll* >( *it ) == 0 ) continue;
    RPCRoll* roll = dynamic_cast< RPCRoll*>( *it );
    addDet(roll);
    
  } // RpcDet loop end
  
  // Separete reference curls from others, should be done in one step
  for(RPCCurlMap::iterator it = m_RPCCurlMap.begin();
      it!=m_RPCCurlMap.end();
      it++)
  {
    if ( (it->second).isRefPlane() ){
      m_refRPCCurlMap[it->first]=it->second;
    }
    else  
      m_otherRPCCurlMap[it->first]=it->second;
  }
  
  // Make links
  /*
  // Pseudo-code for all the action
  for ( RPCCurlMap::iterator itRefCurl=m_refRPCCurlMap.begin(); 
        itRefCurl != m_refRPCCurlMap.end();
        itRefCurl++)//loop over reference curls
  {
    for(;;) //within the refCurl loop over refGlobalStrips
    {
      
        links.push(RefDetId.raw,stripNo,RPCConnection)// RPCConnection should contain tower no, plane no and PAC_no (1...144)
        currentPac = 
        
        for(;;)// Loop over otherCurls
        {
          if( otherCurl contributes to this ref curl){
            for(;;) // loop over otherGlobalStrips
            {
              if (otherGlobalStrip contributes to currentPac)
              {
                
                  links.push(otherGlobalStrip,stripNo,RPCConnection)
                
              }
            }//otherGlobalStrip
        } //otherCurls loop end
      } // refGlopbalStrips loop end
  } // ref Curl's loop end
  //*/
  
  //loop over reference curls
  for ( RPCCurlMap::iterator itRefCurl=m_refRPCCurlMap.begin(); 
        itRefCurl != m_refRPCCurlMap.end();
        //itRefCurl != ++m_refRPCCurlMap.begin();
        itRefCurl++)
  {
    
    
    //(itRefCurl->second).makeConnections(&(itRefCurl->second)); // XXX - clear me!!!
    //loop over other curls
    //*
    for ( RPCCurlMap::iterator itOtherCurl=m_otherRPCCurlMap.begin(); 
          itOtherCurl != m_otherRPCCurlMap.end();
          itOtherCurl++)
    {
      (itRefCurl->second).makeRefConnections(&(itOtherCurl->second));
    
    } //otherCurl loop end
    //*/
  } // refCurl's loop end
  
  
  
  
  m_isGeometryBuilt=true;
  printCurlMapInfo();
}

//#############################################################################
/**
 *
 * \brief Adds detID to the collection
 * \todo Change method to calculate minTower and maxTower (use predefined table)
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
  
  
  for ( RPCCurlMap::iterator it=m_refRPCCurlMap.begin(); it != m_refRPCCurlMap.end(); it++){
    std::cout << "------------------------------"<< std::endl;
    std::cout << "CurlId " << (it->first) << " " << std::endl;
    (it->second).printContents();
  }
  for ( RPCCurlMap::iterator it=m_otherRPCCurlMap.begin(); it != m_otherRPCCurlMap.end(); it++){
    std::cout << "------------------------------"<< std::endl;
    std::cout << "CurlId " << (it->first) << " " << std::endl;
    (it->second).printContents();
  }
  
  std::cout<< "No of refs: " << m_refRPCCurlMap.size() << std::endl;
  std::cout<< "No of others: " << m_otherRPCCurlMap.size() << std::endl;


}
