/** \file RPCCurl.cc
 *
 *  $Date: 2006/06/09 12:35:20 $
 *  $Revision: 1.6 $
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
  
  m_physStripsInCurl = 0;
  m_virtStripsInCurl = 0;
  
  m_isRefPlane = false;
  m_isDataFresh = true;
  m_didVirtuals = false; 
  
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
    
    if (m_globRoll < 0){
      int temp = m_towerMin;
      m_towerMin = -m_towerMax;
      m_towerMax = -temp; // swap is needed (!)
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
 * \brief Makes connections for curls not beeing a refernce ones
 *
 */
//#############################################################################
int RPCCurl::makeOtherConnections(float phiCentre, int tower, int PAC){
  
  if (isRefPlane()){
    std::cout << "Trouble. Curl " << m_curlId
        << " is a reference curl. makeOtherConnections() is not good for reference curls"
        << std::endl;
    return -1;
  }

  if ( (tower < getMinTower()) || (tower > getMaxTower()))  // This curl not contributes to this tower.
    return 0;

  doVirtualStrips();
  
  RPCConnection newConnection;
  newConnection.PAC = PAC;
  newConnection.tower = tower; 
  newConnection.logplane = giveLogPlaneForTower(newConnection.tower);
  
  if (newConnection.logplane < 0){
    std::cout << "Trouble. Curl "<< getCurlId()
        << " wants to contribute to tower " << tower
        << std::endl;
    return -1;
  }
  
  int logplaneSize = LOGPLANE_SIZE[std::abs(newConnection.tower)][newConnection.logplane-1];
  //int logplaneSize = LOGPLANE_SIZE[std::abs(newConnection.tower)][m_hwPlane-1];
  
  if ((logplaneSize > 72)||(logplaneSize < 1)){
    std::cout << "Trouble. Curl "<< getCurlId()
        << " wants to have wrong strips number (" << logplaneSize<< ")"
        << " in plane " << newConnection.logplane
        << " in tower " << newConnection.tower
        << std::endl;
    return -1;
  }
  
  const float pi = 3.141592654;
      
  // \note The  m_stripPhiMap is sorted in a special way (see header)
  GlobalStripPhiMap::const_iterator it = m_stripPhiMap.lower_bound(phiCentre);
  if (it==m_stripPhiMap.end()){
    if (it->first<2*pi){
      it == m_stripPhiMap.begin();
    }
    else{
       std::cout << "Trouble. Phi to big " 
          << "(" << phiCentre << ")"
          << std::endl;
       return -1;
    }
  }
    
  // XXX - possible source of logical errors - cones may be shifted +-1 comparing to ORCA
  for (int i=0; i < logplaneSize/2; i++){
    if (it==m_stripPhiMap.begin())
      it=m_stripPhiMap.end();  // (m_stripPhiMap.end()--) is ok.
    
    it--;
  }
  
  for (int i=0; i < logplaneSize; i++){
    stripCords scTemp = it->second;
    newConnection.posInCone = i+1;
    m_links[it->second].push_back(newConnection);

    it++;
    if (it==m_stripPhiMap.end())
      it=m_stripPhiMap.begin();
  }
  
  return 0;
  
}
//#############################################################################
RPCCurl::RPCLinks RPCCurl::giveConnections(){
  
  return m_links;
}
//#############################################################################
/**
 *
 * \brief Makes connections for reference curls
 * \todo Calculate centrePhi in more elegant way
 * \todo The loop over phi strips should be done once - loop over otherCurls should be here, not in RPCTriggerGeo.
 * \note Conevention: first strip in logplane is no. 1
 * \todo Check if strip numbering convention is same as in ORCA
*
 */
//#############################################################################
int RPCCurl::makeRefConnections(RPCCurl *otherCurl){
  
  if (!isRefPlane()){
    std::cout << "Trouble. Curl " << m_curlId 
        << " is not a reference curl. makeRefConnections() is good only for reference curls"
        << std::endl;
    return -1;
  }
  
  doVirtualStrips();
  
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
        if (plus8==m_stripPhiMap.end()){ // \note The  m_stripPhiMap is sorted in a special way (see header)
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
      otherCurl->makeOtherConnections(centrePhi, m_towerMin, curPacNo);// Make Connections within the other curl
    } // new pac end
    
    RPCConnection newConnection;
    newConnection.PAC = curPacNo;
    newConnection.tower = m_towerMin; // For refCurl m_towerMin and m_towerMax are equal
    newConnection.posInCone = curStripNo-curBegStripNo+1;  // Conevention: first strip in logplane is no. 1
    
    newConnection.logplane = giveLogPlaneForTower(newConnection.tower);
  
    if (newConnection.logplane < 0){
      std::cout << "Trouble. Strip " << it->second.stripNo
      << " of det " << it->second.detRawId
      << " has negative logplane"
      << std::endl;
    }
    
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
      
    } // end check if strip allready in map
    curStripNo++;
  }// end loop over strips
  
  
  return 0;    
}
//#############################################################################
/**
 *
 * \brief Calculates logplane
 * \todo Clean this method
 *
 */
//#############################################################################
int RPCCurl::giveLogPlaneForTower(int tower){
    
  int logplane = -1;
  for (int i=0;i<3;i++){
    int ttemp = mrtow [std::abs(m_globRoll)] [m_hwPlane-1][i];
    if ( ttemp == std::abs(tower) )
      logplane = mrlogp [std::abs(m_globRoll)] [m_hwPlane-1][i];
  }
    


  return logplane;

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
  
  float maxPhi=0, minPhi=0;
  int maxStripNo=0;
  bool firstIt=true;
  
  for (it = sourceMap.begin(); it != sourceMap.end(); it++){

    float phi = it->second;
    stripCords sc;
    sc.stripNo = it->first;
    sc.detRawId = rawId;
    sc.isVirtual = false;
    m_stripPhiMap[phi]=sc;
    
    if(firstIt){
      maxPhi=phi;
      minPhi=phi;
      maxStripNo = sc.stripNo;
      firstIt=false;          
    } 
    
    if(phi < minPhi){
      minPhi=phi;
    }
    if(phi > maxPhi){
      maxPhi=phi;
    }
    
    if (maxStripNo<sc.stripNo)
      maxStripNo=sc.stripNo;
    
    m_physStripsInCurl++;
  }// loop end
  
}
//#############################################################################
/**
 *
 * \brief Fills strip map with virtual strips
 * \bug Curl 4102 has diffrent no. of virtuals than 4002.
 * \todo Improve this function. Some curls seem to have wrong number of virtuals
 * \todo Possible xcheck - check if virtual and physical strips sum to 1152 or more
 * \todo Implement check if we have symetry (x1xx vs x0xx; whell +y vs -y)
 *
 */
//#############################################################################
void RPCCurl::doVirtualStrips(){
  
  if (m_didVirtuals){ // Run once
    return;
  }
  m_didVirtuals=true;
  
  const float pi = 3.141592654;

  bool firstRun=true;
  GlobalStripPhiMap newVirtualStrips;
  
  double dphi=2.0*pi/1152;  // defines angular granulation of strips.
  
  float phiMinNext=0, phiMaxNext=0, phiMinLast=0,phiMaxLast=0;
  uint32_t rawDetIDLast=0,rawDetIDNext=0;
  
  //now we iterate over all dets adding virtual strips begining from phiMax+dphi
  RPCDetInfoPhiMap::const_iterator it;
  for (it=m_RPCDetPhiMap.begin(); it != m_RPCDetPhiMap.end(); it++){
    
    RPCDetInfo *det = &m_RPCDetInfoMap[it->second];
    
    /*
      In first iteration we dont have *Last and *Next values. We must suck them
      in 'artificial' way
    */
    if(firstRun){
      phiMinNext = det->getMinPhi();
      phiMaxNext = det->getMaxPhi();
      rawDetIDNext=it->second; // not really needed
      
      RPCDetInfoPhiMap::const_reverse_iterator itTemp = m_RPCDetPhiMap.rbegin();
      RPCDetInfo *detTemp = &m_RPCDetInfoMap[itTemp->second];
      phiMinLast=detTemp->getMinPhi();
      phiMaxLast=detTemp->getMaxPhi();
      rawDetIDLast= itTemp->second;
      firstRun=false;
    }
    else {
      rawDetIDLast=rawDetIDNext;
      rawDetIDNext=it->second;
      phiMinLast = phiMinNext;
      phiMaxLast = phiMaxNext;
      phiMinNext = det->getMinPhi();
      phiMaxNext = det->getMaxPhi();
    }
    
    float delta = phiMinNext - phiMaxLast;
    delta += 2*pi*(delta<-5);// (-5) Fixes problem of overlaping chambers
    
    if (delta<0)
      continue;
    
    int stripsToAdd = (int)((delta)/dphi+0.5)-1;
    
    stripCords sc;
    sc.detRawId = rawDetIDLast;
    sc.stripNo = 0;
    sc.isVirtual = true;
    for (int i = 0;i<stripsToAdd;i++){
        sc.stripNo--;
        newVirtualStrips[phiMaxLast+dphi*(i+1)]=sc;
        m_virtStripsInCurl++;
    }
  } // loop over dets end 
  
  
  if ( (isRefPlane()) && (m_virtStripsInCurl+m_physStripsInCurl!=1152)){
    std::cout<<"Trouble. Reference curl " << getCurlId() 
        << " has " << m_virtStripsInCurl+m_physStripsInCurl << " strips."
        << std::endl;
  }
  
  m_stripPhiMap.insert(newVirtualStrips.begin(),newVirtualStrips.end() );
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
  else if (m_region == 0 && std::abs(m_ring)==2 && m_hwPlane == 6) // for barell wheel -2,2 refplane is hwPlane=6
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
//## \note Curl of hwPlane = 2, roll = 8 is connected nowhere. Why?
//##
//#############################################################################


// Straigth from ORCA
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


// Straigth from ORCA
const unsigned int RPCCurl::LOGPLANE_SIZE[RPCCurl::TOWERMAX+1][RPCCurl::NHPLANES] = {
//const unsigned int RPCCurl::LOGPLANE_SIZE[17][6] = {
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

  /*
  if (getCurlId()==71001){

    GlobalStripPhiMap::const_iterator it;
    for (it=m_stripPhiMap.begin(); it != m_stripPhiMap.end(); it++){
      std::cout << "phi" << it->first
          << " stripNo=" << (it->second).stripNo
          << " isVirtual=" << (it->second).isVirtual
          <<std::endl;
    }
  }//*/

  if (isRefPlane())
    std::cout<<"+";
  else
    std::cout<<" ";
  
  std::cout << "No. of DetInfo's " << m_RPCDetInfoMap.size()
      << "; towers: min= " << m_towerMin 
      << " max= " << m_towerMax 
      << "|globRoll= " << m_globRoll
      << " hwPlane= " << m_hwPlane
      << "|strips:"
      << " phys= " << m_physStripsInCurl
      << " virt= " << m_virtStripsInCurl
      << " all= " << m_virtStripsInCurl+m_physStripsInCurl
      << "|strips conneced: " << m_links.size() // with or without virtual strips. Check it, it may have changed
      << std::endl;
  
  
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


