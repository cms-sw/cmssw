/** \file RPCTriggerGeo.cc
 *
 *  $Date: 2007/01/30 08:12:56 $
 *  $Revision: 1.18 $
 *  \author Tomasz Fruboes
 */

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "L1Trigger/RPCTrigger/interface/RPCTriggerGeo.h"
#include "L1Trigger/RPCTrigger/interface/RPCException.h"
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
  m_fixRPCGeo = true;

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
 * \note m_Code accessing geometry info is heavly based on
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
    
  //printRingFromRollsMapInfo();
  
  // Free memory
  m_RPCRingFromRollsMap.clear();
  m_refRPCRingFromRollsMap.clear();
  m_otherRPCRingFromRollsMap.clear();
    
  m_isGeometryBuilt=true;
  
}
//#############################################################################
/**
 *
 * \brief Returns vector of tower numbers coresponding to given strip
 *
*/
//#############################################################################
std::vector<int> RPCTriggerGeo::getTowersForStrip(RPCDetId detID, int strip){

  std::vector<int> towersV;
  RPCRingFromRolls::stripCords sc;
  sc.m_detRawId=detID.rawId();
  sc.m_stripNo=strip;
  sc.m_isVirtual=false;

  if (m_links.find(sc)!=m_links.end()){
     RPCRingFromRolls::RPCConnectionsVec stripCons = m_links[sc];
     for (RPCRingFromRolls::RPCConnectionsVec::iterator it = stripCons.begin();
          it != stripCons.end();
          it++)
     {
       towersV.push_back(it->m_tower);
     }  
  }

  else if ( m_detsToIngore.find(sc.m_detRawId)==m_detsToIngore.end() ) { 
    // FIXME: commented out due to changes in handling of overlaping chambers
    //throw RPCException( "Strip not present in map ");
  } 
  
  else {
    // Give output that requested strip is ignored by RPCTrigger

  }


  return towersV;

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

  // This two RingFromRolls werent connected anywhere in ORCA. They are filtered out for consitency with ORCA.
  if ( (detInfo.getRingFromRollsId() == 2108) ||(detInfo.getRingFromRollsId() == 2008) ){
    m_detsToIngore.insert(detInfo.rawId());
    return;
  }
  
  
  if( m_RPCRingFromRollsMap.find(detInfo.getRingFromRollsId()) != m_RPCRingFromRollsMap.end() ){ // RingFromRolls allready in map

     m_RPCRingFromRollsMap[detInfo.getRingFromRollsId()].addDetId(detInfo);

  } else {  // add a new RingFromRolls
    
    RPCRingFromRolls newRingFromRolls;
    newRingFromRolls.fixGeo(m_fixRPCGeo);
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

  std::vector<RPCLogHit> logHits;
    
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
      sc.m_detRawId=rawId;
      sc.m_stripNo=digiIt->strip();
      sc.m_isVirtual=false;
      
      // Find strip in map
      if (m_links.find(sc)!=m_links.end()){
      
        RPCRingFromRolls::RPCConnectionsVec stripCons = m_links[sc];
        //Build loghits
        for (RPCRingFromRolls::RPCConnectionsVec::iterator it = stripCons.begin();
             it != stripCons.end();
             it++)
        {
          logHits.push_back( RPCLogHit(it->m_tower, it->m_PAC, it->m_logplane, it->m_posInCone) );
        }
      
      } 
      // m_detsToIngore fixes problem with two unconnected curls (ORCA consistency)
      else if ( m_detsToIngore.find(rawId)==m_detsToIngore.end() ) { 
//       RPCDetId missing = RPCDetId(sc.m_detRawId);
        // FIXME: commented out due to changes in handling of overlaping chambers
        //throw RPCException( "Strip not present in map ");
//            << missing;
        
        
      }
    } // for digiCollection
  }// for detUnits

  
  // Build cones
  L1RpcLogConesVec ActiveCones;
  
  std::vector<RPCLogHit>::iterator p_lhit;
  for (p_lhit = logHits.begin(); p_lhit != logHits.end(); p_lhit++){
    bool hitTaken = false;
    L1RpcLogConesVec::iterator p_cone;
    for (p_cone = ActiveCones.begin(); p_cone != ActiveCones.end(); p_cone++){
      hitTaken = p_cone->addLogHit(*p_lhit);
      if(hitTaken)
        break;
    }

    if(!hitTaken) {
      RPCLogCone newcone(*p_lhit);
      newcone.setIdx(ActiveCones.size());
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
//#############################################################################
/**
*
* \brief Util function to print m_links map contents
*
*/
//#############################################################################
void RPCTriggerGeo::printLinks(){ 

  for(int iTower=0;iTower<17;iTower++){
    for(int iCone=0;iCone<144;iCone++){
      for(int iPlane=1;iPlane<7;iPlane++){
        if(iTower!=0 || iCone!=1) continue;
        std::cout<<"m_Tower, cone: "<<iTower<<" "<<iCone<<std::endl;
        
        RPCRingFromRolls::RPCLinks::const_iterator CI= m_links.begin();
        for(;CI!=m_links.end();CI++){
          RPCRingFromRolls::stripCords aCoords = CI->first;
          RPCRingFromRolls::RPCConnectionsVec aConnVec = CI->second;
          
          RPCRingFromRolls::RPCConnectionsVec::const_iterator aConnCI = aConnVec.begin();
          for(;aConnCI!=aConnVec.end();aConnCI++){
            if(aConnCI->m_tower==iTower && 
               aConnCI->m_PAC==iCone &&
               aConnCI->m_logplane==iPlane)
            {
              std::cout<<"chId: "<<aCoords.m_detRawId
                       <<" chStrip: "<<aCoords.m_stripNo;
              std::cout<<" m_PAC: "<<aConnCI->m_PAC
                       <<" m_tower: "<<aConnCI->m_tower
                       <<" logPlane: "<<aConnCI->m_logplane
                       <<" m_posInCone: "<<aConnCI->m_posInCone
                       <<std::endl;
              ////////////////////
            }
          }
        }
      }
    }
  }
     
     
}
