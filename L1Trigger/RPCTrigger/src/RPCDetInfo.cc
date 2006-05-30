/** \file RPCDetInfo.cc
 *
 *  $Date: 2006/05/29 12:00:00 $
 *  $Revision: 1.1 $
 *  \author Tomasz Fruboes
 */

#include <cmath>
//#include <iostream>
#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"



///////////////////////////////////////////////////////////////////////////////
/**
 *
 * \brief Construct from roll
 *
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

  m_hwPlane = getHwPlane();

  // Determine min and max \eta values
  
  const StripTopology* topology = dynamic_cast<const StripTopology*>
                                  (&(roll->topology()));
  
  float stripLength = topology->localStripLength(LocalPoint( 0., 0., 0. ));    

  // The list of chamber local positions used to find etaMin and etaMax
  // of a chamber. You can add as many points as desire, but make sure
  // the point lays _inside_ the chamber.
  // Method doesn't work currently :(

  //*  // using y as nonzero doesnt help

  std::vector<Local3DPoint> edges;
  edges.push_back(Local3DPoint( 0., 0., stripLength/2.)); // Add (?) correction for
  edges.push_back(Local3DPoint( 0., 0.,-stripLength/2.)); // nonzero chamber height

  std::vector<float> etas;
  for (unsigned int i=0; i < edges.size(); i++){
    GlobalPoint gp = roll->toGlobal( edges[i] );
    etas.push_back( gp.eta() );
  }

  m_etaMin = *(min_element(etas.begin(), etas.end()));
  m_etaMax = *(max_element(etas.begin(), etas.end()));

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

  /*
  std::cout
      << mHwPlane << " " << mRegion+2 << " "
      << mRing+2 << " " << mRoll << std::endl;
  */

  // Constants are added to have positive numbers      
  int curlId = 1000*(m_region+2) +  // barell is now 2, endcaps are 1 and 3
                100*(m_ring+2) +    // barell may have negative wheel no !                
                 10*(m_hwPlane) +     //1...6
                  1*(m_roll);
                   

  return curlId;
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

  /*
  if ( eta  > 2.15 ) {  // the number is arbitrary but close to real world limit (2.1),
                        // tests consistency of data
    RPCDetId tmpDetId(m_detId);
    std::cout << "Trouble with detId " << m_detId
              << " eta=" << eta
              << " region= " << tmpDetId.m_region()
              << std::endl;
  }
  */
  
  int tower = 0;
  // The highest tower no is 16
  while ( (eta > m_towerBounds[tower]) && (tower!=16) ){
    tower++;
  }


  //std::cout << "eta " << eta << " tower " << tower << std::endl;

  if (sign == 0)
    return -tower;
  else
    return tower;

}

///////////////////////////////////////////////////////////////////////////////
/**
 * \brief sets hardware plane number (mHwPlane)
 * \todo Check layer convention (which is inner/outer) will show up with number of curls beeing reference
 *      
 * \todo Clean this function
*/
///////////////////////////////////////////////////////////////////////////////
int RPCDetInfo::getHwPlane()
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
    hwPlane = station+2;
  } 
  else if ( station == 1 && layer == 1) {
    hwPlane = 1;
  }
  else if ( station == 1 && layer == 2) {
    hwPlane = 2;
  }
  else if ( station == 2 && layer == 1) {
    hwPlane = 3;
  }
  else if ( station == 2 && layer == 2) {
    hwPlane = 4;
  }

  return hwPlane;
  
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
/**
*
* \brief Sets m_etaMin
* \todo check (?) if eta is ok.
*
*/
///////////////////////////////////////////////////////////////////////////////
void RPCDetInfo::setEtaMin(float eta){
  m_etaMin = eta;  
}

///////////////////////////////////////////////////////////////////////////////
/**
*
*  \brief   Sets m_etaMax
*  \todo check (?) if eta is ok.
*
*/
///////////////////////////////////////////////////////////////////////////////
void RPCDetInfo::setEtaMax(float eta){
  m_etaMax = eta;  
}


const float RPCDetInfo::m_towerBounds[] = {0.07, 0.27, 0.44, 0.58, 0.72, 0.83, 0.93, 1.04, 1.14,
                            1.24, 1.36, 1.48, 1.61, 1.73, 1.85, 1.97, 2.10 };
