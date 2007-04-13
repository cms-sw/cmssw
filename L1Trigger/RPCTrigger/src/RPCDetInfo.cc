/** \file RPCDetInfo.cc
 *
 *  $Date: 2007/04/06 10:02:36 $
 *  $Revision: 1.14 $
 *  \author Tomasz Fruboes
 */

#include <cmath>
#include <algorithm>
#include "L1Trigger/RPCTrigger/interface/RPCDetInfo.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"


///////////////////////////////////////////////////////////////////////////////
/**
 *
 * \brief Construct from roll
 * \todo To determine m_towerMin and m_towerMax we should use predefined table. 
 *        Solution with minEta and maxEta is evil :)
*/
///////////////////////////////////////////////////////////////////////////////
RPCDetInfo::RPCDetInfo(RPCRoll* roll){

  RPCDetId detId = roll->id();

  RPCGeomServ grs(detId);
  m_globRoll = grs.eta_partition();

  m_detId = detId.rawId();
  m_region = detId.region(); 

  // Hand;le endcaps.  Needed to verify that etaPartition works ok in barrel
  //if (m_region==1 || m_region == -1){
  //    m_globRoll = 12;
  //}
  
  m_ring = detId.ring();
  m_station = detId.station();
  m_layer = detId.layer();
  m_roll = detId.roll();
  m_sector = detId.sector();
  m_subsector = detId.subsector();

  setHwPlane();

  

  // Determine min and max \eta values
  const StripTopology* topology = dynamic_cast<const StripTopology*>
                                  (&(roll->topology()));
  
  float stripLength = topology->localStripLength(LocalPoint( 0., 0., 0. ));    
  
  // The list of chamber local positions used to find etaMin and etaMax
  // of a chamber. You can add as many points as desire, but make sure
  // the point lays _inside_ the chamber.

  std::vector<LocalPoint> edges;
  edges.push_back(LocalPoint(0., stripLength/2., 0.)); // Add (?) correction for
  edges.push_back(LocalPoint(0.,-stripLength/2., 0.)); // nonzero chamber height

  std::vector<float> m_etas;
  for (unsigned int i=0; i < edges.size(); i++){
    GlobalPoint gp = roll->toGlobal( edges[i] );
    m_etas.push_back( gp.eta() );
  }

  m_etaMin = *(min_element(m_etas.begin(), m_etas.end()));
  m_etaMax = *(max_element(m_etas.begin(), m_etas.end()));
  m_etaCentre =   roll->toGlobal(LocalPoint( 0., 0., 0. )).eta();
  
  m_towerMin = etaToTower(m_etaMin);
  m_towerMax = etaToTower(m_etaMax);
  
  m_phi = transformPhi( (float)(roll->toGlobal(LocalPoint(0., 0., 0.)).phi()) );


  // add strips to the map
  for (int i=1; i<=roll->nstrips(); i++ ) { // note: the strip numbering convention is likely to chnage in future
  
    LocalPoint lStripCentre = roll->centreOfStrip(i);
    GlobalPoint gStripCentre = roll->toGlobal(lStripCentre);
    float phiRaw = gStripCentre.phi();
    float phi = transformPhi( phiRaw );
    
    m_stripPhiMap[i]=phi;
    
  }

  // Fill m_phiMin and m_phiMax values
  LocalPoint lStripCentre = roll->centreOfStrip(1); // XXX - strip numbering convention(!)
  GlobalPoint gStripCentre = roll->toGlobal(lStripCentre);
  float phi1 = transformPhi(gStripCentre.phi());

  LocalPoint lStripCentre2 = roll->centreOfStrip(roll->nstrips());// XXX - strip numbering convention(!)
  GlobalPoint gStripCentre2 = roll->toGlobal(lStripCentre2);
  float phi2 =  transformPhi(gStripCentre2.phi());

 /* std::cout << "Roll: " <<m_globRoll 
            << " plane: "<< m_hwPlane 
            << " z: " << gStripCentre2.z() 
            << " r: " << gStripCentre2.perp() 
            <<std::endl;
*/
  m_phiMin = std::min(phi1,phi2);
  m_phiMax = std::max(phi1,phi2);
  
  // fix the problem around phi=0: (2pi - epsilon) < (0+epsilon)
  if ( (m_phiMin<1)&&(m_phiMax>5) ){ 
    float temp = m_phiMin;
    m_phiMin = m_phiMax;
    m_phiMax = temp;
  }

}
///////////////////////////////////////////////////////////////////////////////
/**
 *
 * \brief gets coresponding curlid.
 * \note the ring is numbered in special way:
 *          - for -endcap ring no 1 is closest to the beam
 *          - for +endcap ring no 1 is farest to the beam
 *          - for barrel ring equals wheel no
 *
*/
///////////////////////////////////////////////////////////////////////////////
int RPCDetInfo::getRingFromRollsId(){
  
  int gr = getGlobRollNo();
  int curlId = 1000*(m_hwPlane) +     //1...6
                100*(etaToSign(gr) ) + //
                  1*( std::abs(getGlobRollNo()) );     //-17...17
  
  return curlId;
}
//#############################################################################
/**
 *
 * \brief Calculates globall iroll no
 *
 */
//#############################################################################
int RPCDetInfo::getGlobRollNo(){


   return m_globRoll;
/*
  int globRoll=20;
    
  if (m_region==0){ //barell
    
    int hr = 0;
    switch (m_roll) {
      case 1:
        hr=1;
        break;
      case 2:
        hr=0;
        break;
      case 3:
        hr=-1;
        break;
    }
    if (m_ring==0) { // Temp fix for eta problem in barell ring0
  	 
  	        // std::cout << m_sector << " " << m_subsector << std::endl;
      if(m_sector ==  1 ||
         m_sector ==  4 || m_sector ==  5 ||
         m_sector ==  8 || m_sector ==  9 ||
         m_sector == 12)
      {
          if (hr==-1)
              hr=1;
          else if (hr==1)
              hr=-1;
      }
    }


    if (m_ring >= 0)
      globRoll = m_ring*3-hr;
    else
      globRoll = m_ring*3+hr;
  } 
  else {
  
    int mr = 0;
    switch (m_ring) {
      case 1:
        mr=3;
        break;
      case 2:
        mr=2;
        break;
      case 3:
        mr=1;
        break;
    }
    globRoll=((mr-1)*3+m_roll+7);
  }
  
  if (m_region==-1)
    globRoll= -globRoll;
  
  if ( (globRoll==20) || (globRoll==-20))
    edm::LogError("RPCTrigger") << "Problem with RPCDetInfo::getGlobRollNo function. GlobRoll=" << globRoll;
    
  return globRoll;*/

}
///////////////////////////////////////////////////////////////////////////////
/**
 *
 * \brief Converts eta to coresponding m_tower number
 * \todo store somewhere MAXTOWER no.
 * \todo store somewhere the max eta value (it will tell us if we properly used geometry)
 *
*/
///////////////////////////////////////////////////////////////////////////////
int RPCDetInfo::etaToTower(float eta){

  int sign = etaToSign(eta);
  eta = std::fabs(eta);

  int m_tower = 0;
  // The highest m_tower no is 16
  while ( (eta > m_towerBounds[m_tower]) && (m_tower!=16) ){
    m_tower++;
  }

  if (sign == 0)
    return -m_tower;
  else
    return m_tower;

}
///////////////////////////////////////////////////////////////////////////////
/**
 * 
 * \brief Returns hardware plane number (mHwPlane)
 * \note Layer convention seems to be ok.
 *
*/
///////////////////////////////////////////////////////////////////////////////
void RPCDetInfo::setHwPlane()
{    
  int region = m_region;
  int station = m_station;
  int layer = m_layer;
    
  int hwPlane = 0;
  if (region != 0){ // endcaps
    hwPlane = station;
  }
  // Now comes the barell
  else if ( station > 2 ){
    hwPlane = station;
  } 
  else if ( station == 1 && layer == 1) {
    hwPlane = 1;
  }
  else if ( station == 1 && layer == 2) {
    hwPlane = 5;
  }
  else if ( station == 2 && layer == 1) {
    hwPlane = 2;
  }
  else if ( station == 2 && layer == 2) {
    hwPlane = 6;
  }

  m_hwPlane = hwPlane;
  
}
///////////////////////////////////////////////////////////////////////////////
/**
*
*  \brief Gives sign of eta (+) -> 1; (-) -> 0
*
*/
///////////////////////////////////////////////////////////////////////////////
int RPCDetInfo::etaToSign(float eta){

  if (eta < 0) return 0;
  return 1;

}
uint32_t RPCDetInfo::rawId(){
  return m_detId;
}
///////////////////////////////////////////////////////////////////////////////
int RPCDetInfo::getMinTower(){ return m_towerMin; }///<Gives the lowest m_tower number
int RPCDetInfo::getMaxTower(){ return m_towerMax; } ///<Gives the highest m_tower number
float RPCDetInfo::getPhi(){ return m_phi; }///<Gets phi of this detid
float RPCDetInfo::getMinPhi(){ return m_phiMin; }///<Gets minPhi of this detid
float RPCDetInfo::getMaxPhi(){ return m_phiMax; }///<Gets maxPhi of this detid
float RPCDetInfo::getEtaCentre(){ return m_etaCentre;}
int RPCDetInfo::getRegion(){ return m_region; }///<Gets region
int RPCDetInfo::getRing(){ return m_ring; }///<Gets ring
int RPCDetInfo::getHwPlane(){ return m_hwPlane; }///<Gets hwplane
int RPCDetInfo::getRoll(){ return m_roll; }///<Gets roll
RPCDetInfo::RPCStripPhiMap RPCDetInfo::getRPCStripPhiMap(){ return m_stripPhiMap;}///<Gets stripMap
///////////////////////////////////////////////////////////////////////////////
/**
 *
 *  \brief Transforms phi
 *
 */
///////////////////////////////////////////////////////////////////////////////
float RPCDetInfo::transformPhi(float phi){

 
  float pi = 3.141592654;
    
  if (phi < 0)
    return phi+2*pi;
  else
    return phi;
  
}
///////////////////////////////////////////////////////////////////////////////
//
//
///////////////////////////////////////////////////////////////////////////////
const float RPCDetInfo::m_towerBounds[] = {0.07, 0.27, 0.44, 0.58, 0.72, 0.83, 0.93, 1.04, 1.14,
                            1.24, 1.36, 1.48, 1.61, 1.73, 1.85, 1.97, 2.10 };
//#############################################################################
/**
*
* \brief prints the contents of a RpcDetInfo.
*
*/
//#############################################################################
void RPCDetInfo::printContents() {
  
  //*
  LogDebug("RPCTrigger")<<"####"<<std::endl;
  LogDebug("RPCTrigger")<< "DetId "<< rawId() << " Centre Phi " << getPhi();
  LogDebug("RPCTrigger")<< " m_Tower min" << m_towerMin << " m_tower max " << m_towerMax;
    
  /*
  RPCStripPhiMap::const_iterator it;
  for (it = m_stripPhiMap.begin(); it != m_stripPhiMap.end(); it++){
        
  LogDebug("RPCTrigger")
        << "Strip: " << it->first << " "
        << "Phi: " << it->second << " "
        
    
  }
  //*/
}
