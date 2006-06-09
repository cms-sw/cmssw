/** \file RPCDetInfo.cc
 *
 *  $Date: 2006/05/31 16:52:58 $
 *  $Revision: 1.4 $
 *  \author Tomasz Fruboes
 */

#include <cmath>
#include <algorithm>
#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"



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

  m_detId = detId.rawId();
  m_region = detId.region(); 
  m_ring = detId.ring();
  m_station = detId.station();
  m_layer = detId.layer();
  m_roll = detId.roll();
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

  std::vector<float> etas;
  for (unsigned int i=0; i < edges.size(); i++){
    GlobalPoint gp = roll->toGlobal( edges[i] );
    etas.push_back( gp.eta() );
  }

  m_etaMin = *(min_element(etas.begin(), etas.end()));
  m_etaMax = *(max_element(etas.begin(), etas.end()));
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
int RPCDetInfo::getCurlId(){
  
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
    if (m_ring > 0)
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
  
  if (globRoll==20)
    std::cout << "Trouble. " << std::endl;
    
  return globRoll;
}
///////////////////////////////////////////////////////////////////////////////
/**
 *
 * \brief Converts eta to coresponding tower number
 * \todo store somewhere MAXTOWER no.
 * \todo store somewhere the max eta value (it will tell us if we properly used geometry)
 *
*/
///////////////////////////////////////////////////////////////////////////////
int RPCDetInfo::etaToTower(float eta){

  int sign = etaToSign(eta);
  eta = std::fabs(eta);

  int tower = 0;
  // The highest tower no is 16
  while ( (eta > m_towerBounds[tower]) && (tower!=16) ){
    tower++;
  }

  if (sign == 0)
    return -tower;
  else
    return tower;

}
///////////////////////////////////////////////////////////////////////////////
/**
 * \brief Returns hardware plane number (mHwPlane)
 * 
 *      
 * \todo Clean this function
 * \note Layer convention seems to be ok.
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
int RPCDetInfo::getMinTower(){ return m_towerMin; }///<Gives the lowest tower number
int RPCDetInfo::getMaxTower(){ return m_towerMax; } ///<Gives the highest tower number
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
* \brief prints the contents of a RpcDetInfo. Commented out, as cout`s are forbidden
*
*/
//#############################################################################
void RPCDetInfo::printContents() {
  
  //*
  std::cout<<"####"<<std::endl;
  std::cout<< "DetId "<< rawId() << " Centre Phi " << getPhi() << std::endl;
  std::cout<< " Tower min" << m_towerMin << " tower max " << m_towerMax << std::endl;
    
  /*
  RPCStripPhiMap::const_iterator it;
  for (it = m_stripPhiMap.begin(); it != m_stripPhiMap.end(); it++){
        
    std::cout
        << "Strip: " << it->first << " "
        << "Phi: " << it->second << " "
        << std::endl;
    
  }
  //*/
}
