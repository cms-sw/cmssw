/** \file RPCCurl.cc
 *
 *  $Date: 2006/05/31 16:52:58 $
 *  $Revision: 1.4 $
 *  \author Tomasz Fruboes
 */
#include "L1Trigger/RPCTrigger/src/RPCCurl.h"
#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"
#include <cmath>
//#include <algorithm>
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
  m_hwPlane = 0;
  m_region = 0;
  m_ring = 0;
  m_roll = 0;
  m_curlId = 0;
  m_globRoll = 0;
  
  m_isRefPlane = false;
  m_isDataFresh = true;
   
}

RPCCurl::~RPCCurl(){ }
//#############################################################################
/**
*
* \brief Adds detId tu the curl
* \todo Implement check if added detInfo  _does_ belong to this RPCCurl
* \todo Check if added detInfo is allready in map
* \todo Implement xcheck if towers are calculated properly
*
*/
//#############################################################################
bool RPCCurl::addDetId(RPCDetInfo detInfo){
  
  if(m_isDataFresh) { // should be done in constructor...
    
//    m_towerMin=detInfo.getMinTower();
    //m_towerMax=detInfo.getMaxTower();
    
    m_hwPlane = detInfo.getHwPlane();
    m_region = detInfo.getRegion();
    m_ring = detInfo.getRing();
    m_roll = detInfo.getRoll();
    m_curlId = detInfo.getCurlId();
    m_globRoll = detInfo.getGlobRollNo();
    
    setRefPlane();
    
    m_towerMin=-1;
    m_towerMax=-1;
    
    for (int i=0; i < 3; i++){
      int ttemp = mrtow [std::abs(m_globRoll)] [m_hwPlane-1][i];
      if (  ((m_towerMin < 0)||(m_towerMax < 0)) && (ttemp >= 0)   ){ 
        m_towerMin = ttemp;
        m_towerMax = ttemp;
      }
      if (ttemp >= 0) {
        if (ttemp < m_towerMin)
          m_towerMin = ttemp;
        if (ttemp > m_towerMax)
          m_towerMax = ttemp;
      }
    }
    
    
    m_isDataFresh=false;
  } 
  /*
  else 
  {
    int min = detInfo.getMinTower();
    int max = detInfo.getMinTower();
    if (min < m_towerMin)
      min = m_towerMin;
    if (max > m_towerMax)
      max = m_towerMax;
  }
  */
  
  updatePhiStripsMap(detInfo);
  
  m_RPCDetInfoMap[detInfo.rawId()]=detInfo; 
  m_RPCDetPhiMap[detInfo.getPhi()]=detInfo.rawId();
 
  
  return true;
}
//#############################################################################
/**
 *
 * \brief Returns conn
 * \todo Check if strips are stored in phi order
 * \todo Calculate centrePhi in more elegant way
 * \bug It seems that in some refCurls we have less strips than they should have. 
 *
 */
//#############################################################################
int RPCCurl::makeConnections(const RPCCurl *otherCurl){

  if (!isRefPlane()){
    std::cout << "Trouble. Curl " << m_curlId 
        << " is not a reference curl. makeConnections() is good only for reference curls"
        << std::endl;
    return -1;
  }
  
  int curPacNo=0;
  int curStripNo=0;
  int curBegStripNo=0;
  GlobalStripPhiMap::const_iterator it;    
  
  // \note The  m_stripPhiMap is sorted in a special way (see header)
  for (it=m_stripPhiMap.begin(); it != m_stripPhiMap.end(); it++){
    
    if (curStripNo%8==0)    // new pac
    {
      curPacNo++;
      curBegStripNo=curStripNo;
      
      GlobalStripPhiMap::const_iterator plus8 = it;    
      for (int i=0;i<7;i++){  // i<7 (!) - there are 8 strips in ref plane !! i<8 would be wrong
        plus8++;
        if (plus8==m_stripPhiMap.end()){
          plus8--;
          break;
        }
      }
      
      float phi1 = it->first;
      float phi2 = plus8->first;
      float centrePhi = (phi1+phi2)/2;
      if (std::min(phi1,phi2) < 1 && std::max(phi1,phi2) > 5)// to avoid (0+2pi)/2 = pi (should be = 0 )
      {
        const float pi = 3.141592654;
        centrePhi -= pi;
        if (centrePhi<0)
          centrePhi += 2*pi;
      }
      // makeOtherConnections();
    } // new pac end
    
    RPCConnection newConnection;
    newConnection.PAC = curPacNo;
    newConnection.tower = m_towerMin; // For refCurl m_towerMin and m_towerMax are equal
    newConnection.posInCone = curStripNo-curBegStripNo;
    
    // Calculate logplane. Table's are straight from ORCA so the method is ugly
    int lpTemp = -1;
    for (int i=0;i<3;i++){
      int ttemp = mrtow [std::abs(m_globRoll)] [m_hwPlane-1][i];
      if ( ttemp == newConnection.tower)
        lpTemp = mrtow [std::abs(m_globRoll)] [m_hwPlane-1][i];
    }
    
    if (lpTemp < 0){
      std::cout << "Trouble. Strip " << it->second.stripNo
            << " of det " << it->second.detRawId
            << " has negative logplane"
            << std::endl;
    }
    newConnection.logplane = lpTemp;
    
    
    if (m_links.find(it->second)==m_links.end() ){// new strip in map
      m_links[it->second].push_back(newConnection);
    } 
    else {  // strip allready in map, we should have the same connections
      
      RPCConnectionsVec existingConnection = m_links[it->second];
      if ( (existingConnection[0].PAC != newConnection.PAC ) ||  
            (existingConnection[0].tower != newConnection.tower ) ||
            (existingConnection[0].logplane != newConnection.logplane ) ||
            (existingConnection[0].posInCone != newConnection.posInCone ) )
      {
        std::cout << "Trouble. Strip " << it->second.stripNo
            << " of reference det " << it->second.detRawId
            << " has multiple connections"
            << std::endl;
      }
      
    }
    
    
    curStripNo++;
  }
  
  
  return 0;    
}
//#############################################################################
/**
 *
 * \brief Updates m_stripPhiMap
 *
 */
//#############################################################################
void RPCCurl::updatePhiStripsMap(RPCDetInfo detInfo){
  
  uint32_t rawId = detInfo.rawId();
  
  RPCDetInfo::RPCStripPhiMap sourceMap = detInfo.getRPCStripPhiMap();
  RPCDetInfo::RPCStripPhiMap::const_iterator it;
  
  for (it = sourceMap.begin(); it != sourceMap.end(); it++){
    float phi = it->second;
    stripCords sc;
    sc.stripNo = it->first;
    sc.detRawId = rawId;
    m_stripPhiMap[phi]=sc;
  }

}
//#############################################################################
/**
 *
 * \brief Checks and sets value of m_isReferencePlane
 * \note hwPlane numbering is non trivial
 *
 */
//#############################################################################
void RPCCurl::setRefPlane() {
  
  m_isRefPlane = false;
  if (m_region == 0 && std::abs(m_ring)<2 && m_hwPlane == 2) // for barell wheel -1,0,1 refplane is hwPlane=2
    m_isRefPlane = true;
  else if (m_region == 0 && std::abs(m_ring)==2 && m_hwPlane == 6) // for barell wheel -2,0,2 refplane is hwPlane=6
    m_isRefPlane = true;
  else if (m_region != 0 && m_hwPlane == 2) // for endcaps
    m_isRefPlane = true;
  
  if( m_curlId == 2008 || m_curlId == 2108) //exception: endcaps;hwplane 2;farest roll from beam
    m_isRefPlane = false;
}

//#############################################################################
//#
//# Simple getters
//#
//#############################################################################
bool RPCCurl::isRefPlane() const { return m_isRefPlane;} ///< Returns value of m_isReferencePlane
int RPCCurl::getMinTower() const { return m_towerMin;} ///< Returns value of min tower
int RPCCurl::getMaxTower() const{ return m_towerMax;} ///< Returns value of max tower
int RPCCurl::getCurlId() const{ return m_curlId;} ///< Returns value of max tower
//#############################################################################
//##
//##
//##
//#############################################################################
const int RPCCurl::mrtow [RPCCurl::IROLL_MAX+1] [RPCCurl::NHPLANES] [RPCCurl::NPOS] =
//const int RPCCurl::mrtow [] [] [] =
{
// MB1in/MF1  MB2in/MF2   MB3/MF3   MB4/MF4      MB1out    MB2out
//     1          2          3         4           5         6
  { {-1,-1,-1},{ 0,-1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1,-1} },   //roll 0
  { { 0, 1,-2},{ 1,-1,-1},{ 0, 1,-1},{ 0, 1,-1},{ 0, 1,-2},{ 0, 1,-1} },   //     1
  { { 2, 3, 4},{ 2,-1,-1},{ 1, 2,-1},{ 1, 2,-1},{ 2, 3, 4},{ 2, 3,-1} },   //     2  //<<corrected plane 3, kb
  { {-1,-1,-1},{ 3,-1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1,-1} },   //     3
  { { 4, 5,-6},{ 4,-1,-1},{ 2, 3, 4},{ 2, 3,-1},{ 4, 5,-1},{ 3, 4,-1} },   //     4  //<<corrected plane 3, kb
  { { 6, 7, 8},{ 5,-6, 6},{ 4, 5,-1},{ 4,-1,-1},{ 5, 6, 7},{ 5,-1,-1} },   //     5
  { {-1,-1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1,-1},{ 6,-1,-1} },   //     6
  { { 8, 9,-1},{ 7,-7,-1},{ 5, 6,-1},{ 4, 5,-1},{ 7, 8,-1},{ 7,-1,-1} },   //     7
  { {-1, 7,-1},{-1,-1,-1},{ 9,-9,-1},{10,-1,-1},{-1,-1,-1},{-1,-1,-1} },   //     8
  { { 7, 8,-1},{ 8,-1,-1},{ 9,-9,10},{10,11,-1},{-1,-1,-1},{-1,-1,-1} },   //     9
  { { 8,-1,-1},{ 9,-1,-1},{10,11,-1},{11,12,-1},{-1,-1,-1},{-1,-1,-1} },   //    10
  { {10,-1,-1},{10,-1,-1},{11,12,-1},{12,13,-1},{-1,-1,-1},{-1,-1,-1} },   //    11
  { {10,11,-1},{11,-1,-1},{12,13,-1},{13,14,-1},{-1,-1,-1},{-1,-1,-1} },   //    12
  { {11,12,-1},{12,-1,-1},{13,14,-1},{14,15,-1},{-1,-1,-1},{-1,-1,-1} },   //    13
  { {13,14,-1},{13,-1,-1},{14,15,-1},{15,16,-1},{-1,-1,-1},{-1,-1,-1} },   //    14
  { {14,15,-1},{14,-1,-1},{15,16,-1},{16,-1,-1},{-1,-1,-1},{-1,-1,-1} },   //    15
  { {15,16,-1},{15,-1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1,-1} },   //    16
  { {16,-1,-1},{16,-1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1,-1},{-1,-1,-1} }    //    17
};

const int RPCCurl::mrlogp [RPCCurl::IROLL_MAX+1] [RPCCurl::NHPLANES] [RPCCurl::NPOS] =
//const int RPCCurl::mrlogp [] [] [] =
{
// MB1in/MF1  MB2in/MF2   MB3/MF3   MB4/MF4      MB1out    MB2out
//     1          2          3          4          5          6
  {  {0,0,0},   {3,0,0},   {0,0,0},   {0,0,0},   {0,0,0},   {0,0,0} },  // roll  0
  {  {1,1,-1},  {3,0,0},   {5,5,0},   {6,6,0},   {2,2,-2},  {4,4,0} },  //       1
  {  {1,1,1},   {3,0,0},   {5,5,0},   {6,6,0},   {2,2,2},   {4,4,0} },  //       2 <<3
  {  {0,0,0},   {3,0,0},   {0,0,0},   {0,0,0},   {0,0,0},   {0,0,0} },  //       3
  {  {1,1,-5},  {3,0,0},   {5,5,5},   {6,6,0},   {2,2,0},   {4,4,0} },  //       4 <<3
  {  {1,1,1},   {3,-3,3},  {5,5,0},   {6,0,0},   {2,2,2},   {4,0,0} },  //       5
  {  {0,0,0},   {0,0,0},   {0,0,0},   {0,0,0},   {0,0,0},   {4,0,0} },  //       6
  {  {1,1,0},   {3,-3,0},  {5,5,0},   {6,6,0},   {2,2,0},   {4,0,0} },  //       7
  {  {0,5,0},   {0,0,0},   {3,-5,0},  {4,0,0},   {0,0,0},   {0,0,0} },  //       8
  {  {5,3,0},   {4,0,0},   {3,-5,3},  {4,4,0},   {0,0,0},   {0,0,0} },  //       9
  {  {3,0,0},   {2,0,0},   {3,3,0},   {4,4,0},   {0,0,0},   {0,0,0} },  //      10
  {  {1,0,0},   {2,0,0},   {3,3,0},   {4,4,0},   {0,0,0},   {0,0,0} },  //      11
  {  {1,1,0},   {2,0,0},   {3,3,0},   {4,4,0},   {0,0,0},   {0,0,0} },  //      12
  {  {1,1,0},   {2,0,0},   {3,3,0},   {4,4,0},   {0,0,0},   {0,0,0} },  //      13
  {  {1,1,0},   {2,0,0},   {3,3,0},   {4,4,0},   {0,0,0},   {0,0,0} },  //      14
  {  {1,1,0},   {2,0,0},   {3,3,0},   {4,0,0},   {0,0,0},   {0,0,0} },  //      15
  {  {1,1,0},   {2,0,0},   {0,0,0},   {0,0,0},   {0,0,0},   {0,0,0} },  //      16
  {  {1,0,0},   {2,0,0},   {0,0,0},   {0,0,0},   {0,0,0},   {0,0,0} }   //      17
};
//const unsigned int RPCCurl::LOGPLANE_SIZE[RPCCurl::TOWER_COUNT][RPCCurl::LOGPLANES_COUNT] = {
const unsigned int RPCCurl::LOGPLANE_SIZE[17][6] = {
 //LOGPLANE  1,  2,  3   4   5   6
           {72, 56,  8, 40, 40, 24}, //TOWER 0
           {72, 56,  8, 40, 40, 24}, //TOWER 1
           {72, 56,  8, 40, 40, 24}, //TOWER 2
           {72, 56,  8, 40, 40, 24}, //TOWER 3
           {72, 56,  8, 40, 40, 24}, //TOWER 4
           {72, 56, 40,  8, 40, 24}, //TOWER 5
           {56, 72, 40,  8, 24,  0}, //TOWER 6
           {72, 56, 40,  8, 24,  0}, //TOWER 7
           {72, 24, 40,  8,  0,  0}, //TOWER 8
           {72,  8, 40,  0,  0,  0}, //TOWER 9
           {72,  8, 40, 24,  0,  0}, //TOWER 10
           {72,  8, 40, 24,  0,  0}, //TOWER 11
           {72,  8, 40, 24,  0,  0}, //TOWER 12
           {72,  8, 40, 24,  0,  0}, //TOWER 13
           {72,  8, 40, 24,  0,  0}, //TOWER 14
           {72,  8, 40, 24,  0,  0}, //TOWER 15
           {72,  8, 40, 24,  0,  0}  //TOWER 16

};


//#############################################################################
/**
*
* \brief prints the contents of a RPCCurl. Commented out, as cout`s are forbidden
*
*/
//#############################################################################
void RPCCurl::printContents() {
  
  //*
  //std::cout << " ---------------------------------------------------" << std::endl;
  if (isRefPlane())
    std::cout<<" + ";
  else
    std::cout<<"   ";
  
  std::cout << " No. of RPCDetInfo's " << m_RPCDetInfoMap.size()
  //    << "; towers: min= " << m_towerMin 
  //    << " max= " << m_towerMax 
      << " globRoll= " << m_globRoll
      << " hwPlane= " << m_hwPlane
      << " connections: " << m_links.size()
      << std::endl;;
  
  
  /*
  RPCDetInfoPhiMap::const_iterator it;
  for (it = m_RPCDetPhiMap.begin(); it != m_RPCDetPhiMap.end(); it++){
  
    //std::cout
    //    << "Phi: " << it->first << " "
    //   << "detId: " << it->second  << std::endl;
  
    
    m_RPCDetInfoMap[it->second].printContents();
        
  }
  //*/
  /*
  GlobalStripPhiMap::const_iterator it;
  for (it = m_stripPhiMap.begin(); it != m_stripPhiMap.end(); it++){
    std::cout
        << "Phi: " << it->first 
        << " detId: " << it->second.detRawId 
        << " stripNo: " << it->second.stripNo  << std::endl;
  }//*/

}


