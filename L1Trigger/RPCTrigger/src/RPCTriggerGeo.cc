/** \file RPCTriggerGeo.cc
 *
 *  $Date: 2006/07/27 08:57:33 $
 *  $Revision: 1.12 $
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
bool RPCTriggerGeo::isGeometryBuilt(){ return m_isGeometryBuilt; }
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
  
  // Separete reference curls from others, should be done in previous step
  for(RPCRingFromRollsMap::iterator it = m_RPCRingFromRollsMap.begin();
      it!=m_RPCRingFromRollsMap.end();
      it++)
  {
    if ( (it->second).isRefPlane() ){
      m_refRPCRingFromRollsMap[it->first]=it->second;
    }
    else  
      m_otherRPCRingFromRollsMap[it->first]=it->second;
  }
  
  //loop over reference curls
  for ( RPCRingFromRollsMap::iterator itRefRingFromRolls=m_refRPCRingFromRollsMap.begin(); 
        itRefRingFromRolls != m_refRPCRingFromRollsMap.end();
        itRefRingFromRolls++)
  {
    //loop over other curls
    for ( RPCRingFromRollsMap::iterator itOtherRingFromRolls=m_otherRPCRingFromRollsMap.begin(); 
          itOtherRingFromRolls != m_otherRPCRingFromRollsMap.end();
          itOtherRingFromRolls++)
    {
      (itRefRingFromRolls->second).makeRefConnections(&(itOtherRingFromRolls->second));
    } //otherRingFromRolls loop end
  } // refRingFromRolls's loop end
  
  
  // Copy all stripConections into one place
  for ( RPCRingFromRollsMap::iterator it=m_refRPCRingFromRollsMap.begin(); 
        it != m_refRPCRingFromRollsMap.end();
        it++){
          
          RPCRingFromRolls::RPCLinks linksTemp = (it->second).giveConnections();
          m_links.insert(linksTemp.begin(),linksTemp.end() );
          
        }
  
  for ( RPCRingFromRollsMap::iterator it=m_otherRPCRingFromRollsMap.begin(); 
        it != m_otherRPCRingFromRollsMap.end();
        it++){
          
          RPCRingFromRolls::RPCLinks linksTemp = (it->second).giveConnections();
          m_links.insert(linksTemp.begin(),linksTemp.end() );
  
  }
    
  // Free memory
  m_RPCRingFromRollsMap.clear();
  m_refRPCRingFromRollsMap.clear();
  m_otherRPCRingFromRollsMap.clear();
    
  m_isGeometryBuilt=true;
  //printRingFromRollsMapInfo();
}

//#############################################################################
/**
 *
 * \brief Adds detID to the collection
 *
*/
//#############################################################################
void RPCTriggerGeo::addDet(RPCRoll* roll){

  RPCDetInfo detInfo(roll);

  // This two curls werent connected anywhere in ORCA. They are filtered out for consitency with ORCA.
  if ( (detInfo.getRingFromRollsId() == 2108) ||(detInfo.getRingFromRollsId() == 2008) ){
    m_detsToIngore.insert(detInfo.rawId());
    return;
  }
  
  
  if( m_RPCRingFromRollsMap.find(detInfo.getRingFromRollsId()) != m_RPCRingFromRollsMap.end() ){ // RingFromRolls allready in map

     m_RPCRingFromRollsMap[detInfo.getRingFromRollsId()].addDetId(detInfo);

  } else {  // add a new curl
    
    RPCRingFromRolls newRingFromRolls;
    newRingFromRolls.addDetId(detInfo);
    m_RPCRingFromRollsMap[detInfo.getRingFromRollsId()]=newRingFromRolls;

  }

}
//#############################################################################
/**
 *
 * \brief Builds cones from digis
 * \note Based on L1RpcConeBuilder from ORCA
 *
 */
//#############################################################################
L1RpcLogConesVec RPCTriggerGeo::getCones(edm::Handle<RPCDigiCollection> rpcDigis){

  std::vector<L1RpcLogHit> logHits;
    
// Build cones from digis
  RPCDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=rpcDigis->begin();
       detUnitIt!=rpcDigis->end();
       ++detUnitIt)
  {
    const RPCDetId& id = (*detUnitIt).first;
    
    int rawId = id.rawId();
    
    const RPCDigiCollection::Range& range = (*detUnitIt).second;
    
    for (RPCDigiCollection::const_iterator digiIt = range.first;
         digiIt!=range.second;
         ++digiIt)
    {
      RPCRingFromRolls::stripCords sc;
      sc.detRawId=rawId;
      sc.stripNo=digiIt->strip();
      sc.isVirtual=false;
      
      // Find strip in map
      if (m_links.find(sc)!=m_links.end()){
      
        RPCRingFromRolls::RPCConnectionsVec stripCons = m_links[sc];
        //Build loghits
        for (RPCRingFromRolls::RPCConnectionsVec::iterator it = stripCons.begin();
             it != stripCons.end();
             it++)
        {
          logHits.push_back( L1RpcLogHit(it->tower, it->PAC, it->logplane, it->posInCone) );
        }
      
      } 
      // m_detsToIngore fixes problem with two unconnected curls (ORCA consistency)
      else if ( m_detsToIngore.find(rawId)==m_detsToIngore.end() ) { 
      // should throw exception
        RPCDetId missing = RPCDetId(sc.detRawId);
        edm::LogError("RPCTrigger")
            << "Strip " << sc.stripNo 
            << " of det " << sc.detRawId
            << " not present in map "
            << missing;
        
        
      }
    } // for digiCollection
  }// for detUnits

  
  // Build cones
  L1RpcLogConesVec ActiveCones;
  
  std::vector<L1RpcLogHit>::iterator p_lhit;
  for (p_lhit = logHits.begin(); p_lhit != logHits.end(); p_lhit++){
    bool hitTaken = false;
    L1RpcLogConesVec::iterator p_cone;
    for (p_cone = ActiveCones.begin(); p_cone != ActiveCones.end(); p_cone++){
      hitTaken = p_cone->AddLogHit(*p_lhit);
      if(hitTaken)
        break;
    }

    if(!hitTaken) {
      L1RpcLogCone newcone(*p_lhit);
      newcone.SetIdx(ActiveCones.size());
      ActiveCones.push_back(newcone);
    }
  }// for loghits
  
  return ActiveCones;
}
//#############################################################################
/**
*
* \brief Util function to print rpcChambersMap contents
*
*/
//#############################################################################
void RPCTriggerGeo::printRingFromRollsMapInfo(){ // XXX - Erase ME
  
  //*
  for ( RPCRingFromRollsMap::iterator it=m_refRPCRingFromRollsMap.begin(); it != m_refRPCRingFromRollsMap.end(); it++){
    LogDebug("RPCTrigger") << "------------------------------";
    LogDebug("RPCTrigger")  << "RingFromRollsId " << (it->first);
    (it->second).printContents();
  }
  for ( RPCRingFromRollsMap::iterator it=m_otherRPCRingFromRollsMap.begin(); it != m_otherRPCRingFromRollsMap.end(); it++){
    LogDebug("RPCTrigger")<< "------------------------------";
    LogDebug("RPCTrigger") << "RingFromRollsId " << (it->first);
    (it->second).printContents();
  }
  //*/
  
  LogDebug("RPCTrigger") << "No of refs: " << m_refRPCRingFromRollsMap.size();
  LogDebug("RPCTrigger")  << m_otherRPCRingFromRollsMap.size(); 
  LogDebug("RPCTrigger")  << m_links.size();

}
