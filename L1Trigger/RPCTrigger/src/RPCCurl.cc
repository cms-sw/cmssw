/** \file RPCCurl.cc
 *
 *  $Date: 2006/05/30 18:48:40 $
 *  $Revision: 1.3 $
 *  \author Tomasz Fruboes
 */


#include "L1Trigger/RPCTrigger/src/RPCCurl.h"
#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"
//#############################################################################
/**
 *
 * \brief Default constructor
 *
 */
//#############################################################################
RPCCurl::RPCCurl()
{ 
  m_towerMin = 0;
  m_towerMax = 0;
  m_hardwarePlane = 0;
  m_region = 0;
  m_ring = 0;
  m_roll = 0;
  
  m_isDataFresh = true;
}

RPCCurl::~RPCCurl(){ }
//#############################################################################
/**
*
* \brief Adds detId tu the curl
* \todo Implement check if added detInfo  _does_ belong to this RPCCurl
* \todo Check if added detInfo is allready in map
*
*/
//#############################################################################
bool RPCCurl::addDetId(RPCDetInfo detInfo){
  
  if(m_isDataFresh) {
    m_towerMin=detInfo.getMinTower();
    m_towerMax=detInfo.getMaxTower();
    
    m_hardwarePlane = detInfo.getHwPlane();
    m_region = detInfo.getRegion();
    m_ring = detInfo.getRing();
    m_roll = detInfo.getRoll();
    m_isDataFresh=false;
  } 
  else 
  {
    int min = detInfo.getMinTower();
    int max = detInfo.getMinTower();
    if (min < m_towerMin)
      min = m_towerMin;
    if (max > m_towerMax)
      max = m_towerMax;
  }
    
  m_RPCDetInfoMap[detInfo.rawId()]=detInfo; 
  m_RPCDetPhiMap[detInfo.getPhi()]=detInfo.rawId();

  return true;
}

//#############################################################################
/**
*
* \brief prints the contents of a RPCurl. Commented out, as cout`s are forbidden
*
*/
//#############################################################################
void RPCCurl::printContents() {
  
  //*
  //std::cout << " ---------------------------------------------------" << std::endl;
  std::cout << " No. of RPCDetInfo's " << m_RPCDetInfoMap.size()
            << std::endl;
  std::cout << "Tower: min=" << m_towerMin << " max=" << m_towerMax << std::endl;
  
  /*
  RPCDetInfoPhiMap::const_iterator it;
  for (it = m_RPCDetPhiMap.begin(); it != m_RPCDetPhiMap.end(); it++){
        
    std::cout
        << "Phi: " << it->first << " "
        << "detId: " << it->second  << std::endl;
    
    m_RPCDetInfoMap[it->second].printContents();
        
  }
  //*/
}

