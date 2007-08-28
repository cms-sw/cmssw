/** \file RPCRingFromRolls.cc
 *
 *  $Date: 2007/04/13 16:40:10 $
 *  $Revision: 1.8 $
 *  \author Tomasz Fruboes
 */
#include "L1Trigger/RPCTrigger/interface/RPCRingFromRolls.h"
#include "L1Trigger/RPCTrigger/interface/RPCDetInfo.h"
#include "L1Trigger/RPCTrigger/interface/RPCException.h"
#include <cmath>
#include <algorithm>
//#############################################################################
/**
 *
 * \brief Default constructor
 *
 */
//#############################################################################
RPCRingFromRolls::RPCRingFromRolls()
{ 
  m_towerMin = 0;
  m_towerMax = 0;
  m_hwPlane = 0;
  m_region = 0;
  m_ring = 0;
  m_roll = 0;
  m_curlId = 0;
  m_globRoll = 0;
  
  m_physStripsInRingFromRolls = 0;
  m_virtStripsInRingFromRolls = 0;
  
  m_isRefPlane = false;
  m_isDataFresh = true;
  m_didVirtuals = false; 
  m_didFiltering = false;
  
}

RPCRingFromRolls::~RPCRingFromRolls(){ }
//#############################################################################
/**
*
* \brief Adds detId tu the curl
* \todo Implement check if added detInfo  _does_ belong to this RPCRingFromRolls
* \todo check if added detInfo is allready in map
* \todo Implement xcheck if towers are calculated properly
*
*/
//#############################################################################
bool RPCRingFromRolls::addDetId(RPCDetInfo detInfo){
  
  if(m_isDataFresh) { // should be done in constructor...
    
//    m_towerMin=detInfo.getMinTower();
    //m_towerMax=detInfo.getMaxTower();
    
    m_hwPlane = detInfo.getHwPlane();
    m_region = detInfo.getRegion();
    m_ring = detInfo.getRing();
    m_roll = detInfo.getRoll();
    m_curlId = detInfo.getRingFromRollsId();
    m_globRoll = detInfo.getGlobRollNo();
    
    setRefPlane();
    
    m_towerMin=-1;
    m_towerMax=-1;
    
    for (int i=0; i < 3; i++){
      int ttemp = m_mrtow [std::abs(m_globRoll)] [m_hwPlane-1][i];
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
 * \brief Makes connections for Rings not beeing a refernce ones
 *
 */
//#############################################################################
int RPCRingFromRolls::makeOtherConnections(float phiCentre, int m_tower, int m_PAC){
  
  if (isRefPlane()){
    edm::LogError("RPCTrigger") << "Trouble. RingFromRolls " << m_curlId
        << " is a reference curl. makeOtherConnections() is not good for reference curls";
    return -1;
  }

  if ( (m_tower < getMinTower()) || (m_tower > getMaxTower()))  // This curl not contributes to this m_tower.
    return 0;

  doVirtualStrips();
  filterMixedStrips(); // Fixes overlaping chambers problem
  
  RPCConnection newConnection;
  newConnection.m_PAC = m_PAC;
  newConnection.m_tower = m_tower; 
  newConnection.m_logplane = giveLogPlaneForTower(newConnection.m_tower);
    
  if (newConnection.m_logplane < 0){
    
    LogDebug("RPCTrigger") << "Trouble. RingFromRolls "<< getRingFromRollsId()
        << " wants to contribute to m_tower " << m_tower;
       
    return -1;
  }
  
  int logplaneSize = m_LOGPLANE_SIZE[std::abs(newConnection.m_tower)][newConnection.m_logplane-1];
  //int logplaneSize = m_LOGPLANE_SIZE[std::abs(newConnection.m_tower)][m_hwPlane-1];
  
  if ((logplaneSize > 72)||(logplaneSize < 1)){
    LogDebug("RPCTrigger") << "Trouble. RingFromRolls "<< getRingFromRollsId()
        << " wants to have wrong strips number (" << logplaneSize<< ")"
        << " in plane " << newConnection.m_logplane
        << " in m_tower " << newConnection.m_tower;
        
    return -1;
  }
  
      
  // \note The  m_stripPhiMap is sorted in a special way (see header)
  GlobalStripPhiMap::const_iterator it = m_stripPhiMap.lower_bound(phiCentre);
  if (it==m_stripPhiMap.end()){
      it == m_stripPhiMap.begin();
  }
    
  
  for (int i=0; i < logplaneSize/2; i++){ 
    if (it==m_stripPhiMap.begin())
      it=m_stripPhiMap.end();  // (m_stripPhiMap.end()--) is ok.
    it--;
  }
  
  // In barell station 4 (farrest station) chambers overlap in phi.
  // This is the q&d method to avoid mixing of strips in logplanes
  /*
  if (m_region == 0 && m_hwPlane == 4){
               
      std::map<uint32_t,GlobalStripPhiMap> chambersMap;
      std::map<uint32_t,float> lowestPhiMap; // For finding lowest phi in each chamber
      
      for (int i=0; i < logplaneSize; i++){
          stripCords scTemp = it->second;
          (chambersMap[scTemp.m_detRawId])[it->first]=it->second;
          //aMap[it->first]=it->second;
          
          if (lowestPhiMap.find(scTemp.m_detRawId)==lowestPhiMap.end())// New detID
            {
               lowestPhiMap[scTemp.m_detRawId]=it->first;                                               
            } 
          else // detId allready in map
            {
               RPCRingFromRolls::phiMapCompare compare;
                  // compare(a,b) <=>  (a<b)
               if (compare(lowestPhiMap[scTemp.m_detRawId],it->first))
                 {
                   lowestPhiMap[scTemp.m_detRawId]=it->first;
                 }
            }
        it++;
        if (it==m_stripPhiMap.end())
          it=m_stripPhiMap.begin();
      } // for (int i=0; i < logplaneSize; i++) ends
      

      
      // sort chambers in phi
      std::map<float,uint32_t,phiMapCompare> chambersIds;
      
      std::map<uint32_t,float>::const_iterator phiIt =  lowestPhiMap.begin();
      for (;phiIt!=lowestPhiMap.end();phiIt++){
          chambersIds[phiIt->second]=phiIt->first;
      }
      
      // Now we can iterate over the strips in each chamber      
      std::map<float,uint32_t,phiMapCompare>::const_iterator chambersIt = chambersIds.begin();
      int curStripInConeNo = 0;
      for (;chambersIt!=chambersIds.end();chambersIt++){
          GlobalStripPhiMap aMap = chambersMap[chambersIt->second];
          
          GlobalStripPhiMap::const_iterator stripIt = aMap.begin();
          for(;stripIt!=aMap.end();stripIt++){
             //stripCords scTemp = stripIt->second;
             newConnection.m_posInCone = curStripInConeNo;
             m_links[stripIt->second].push_back(newConnection);
             curStripInConeNo++;
          }
      }
  
}
  else // Normal, non overlaping chamber
  {
      for (int i=0; i < logplaneSize; i++){
        stripCords scTemp = it->second;
        newConnection.m_posInCone = i;
        m_links[it->second].push_back(newConnection);
    
        it++;
        if (it==m_stripPhiMap.end())
          it=m_stripPhiMap.begin();
      }
  }
  */
  
  // 
  for (int i=0; i < logplaneSize; i++){
    stripCords scTemp = it->second;
    newConnection.m_posInCone = i;
    m_links[it->second].push_back(newConnection);
    
    it++;
    if (it==m_stripPhiMap.end())
      it=m_stripPhiMap.begin();
  }

    
  return 0;
  
}
//#############################################################################
RPCRingFromRolls::RPCLinks RPCRingFromRolls::giveConnections(){
  
  return m_links;
}
//#############################################################################
/**
 *
 * \brief Makes connections for reference rings
 * \todo Calculate centrePhi in more elegant way
 * \note Conevention: first strip in m_logplane is no. 0
*
 */
//#############################################################################
int RPCRingFromRolls::makeRefConnections(RPCRingFromRolls *otherRingFromRolls){
  
  if (!isRefPlane()){
    
    edm::LogError("RPCTrigger") << "Trouble. RingFromRolls " << m_curlId 
        << " is not a reference curl. makeRefConnections() is good only for reference curls";
        
    return -1;
  }
  
  doVirtualStrips();
  
  int curPacNo=-1;  // pacs are numbered from 0 to 143 (there is curPacNo++ in first iteration)
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
      otherRingFromRolls->makeOtherConnections(centrePhi, m_towerMin, curPacNo);// Make Connections within the other curl
    } // new pac end
    
    RPCConnection newConnection;
    newConnection.m_PAC = curPacNo;
    newConnection.m_tower = m_towerMin; // For refRingFromRolls m_towerMin and m_towerMax are equal
    newConnection.m_posInCone = curStripNo-curBegStripNo;  // Conevention: first strip in m_logplane is no. 0 
    
    newConnection.m_logplane = giveLogPlaneForTower(newConnection.m_tower);
  
    if (newConnection.m_logplane < 0){
      
      edm::LogError("RPCTrigger") << "Trouble. Strip " << it->second.m_stripNo
          << " of det " << it->second.m_detRawId
          << " has negative m_logplane";
      
    }
    
    if (m_links.find(it->second)==m_links.end() ){// new strip in map
      m_links[it->second].push_back(newConnection);
    } 
    else {  // strip allready in map, we should have the same connections
      
      RPCConnectionsVec existingConnection = m_links[it->second];
      if ( (existingConnection[0].m_PAC != newConnection.m_PAC ) ||  
            (existingConnection[0].m_tower != newConnection.m_tower ) ||
            (existingConnection[0].m_logplane != newConnection.m_logplane ) ||
            (existingConnection[0].m_posInCone != newConnection.m_posInCone ) )
      {
        
        edm::LogError("RPCTrigger") << "Trouble. Strip " << it->second.m_stripNo
            << " of reference det " << it->second.m_detRawId
            << " has multiple connections";
            
      }
      
    } // end check if strip allready in map
    curStripNo++;
  }// end loop over strips
  
  
  return 0;    
}
//#############################################################################
/**
 *
 * \brief Calculates m_logplane
 * \todo Clean this method
 *
 */
//#############################################################################
int RPCRingFromRolls::giveLogPlaneForTower(int tower){
    
  int logplane = -1;
  for (int i=0;i<3;i++){
    int ttemp = m_mrtow [std::abs(m_globRoll)] [m_hwPlane-1][i];
    if ( ttemp == std::abs(tower) )
      logplane = m_mrlogp [std::abs(m_globRoll)] [m_hwPlane-1][i];
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
void RPCRingFromRolls::updatePhiStripsMap(RPCDetInfo detInfo){
  
  uint32_t rawId = detInfo.rawId();
  
  RPCDetInfo::RPCStripPhiMap sourceMap = detInfo.getRPCStripPhiMap();
  RPCDetInfo::RPCStripPhiMap::const_iterator it;
  
  float maxPhi=0, minPhi=0;
  int maxStripNo=0;
  bool firstIt=true;
  
  for (it = sourceMap.begin(); it != sourceMap.end(); it++){

    float phi = it->second;
    stripCords sc;
    sc.m_stripNo = it->first;
    sc.m_detRawId = rawId;
    sc.m_isVirtual = false;
    m_stripPhiMap[phi]=sc;
    
    if(firstIt){
      maxPhi=phi;
      minPhi=phi;
      maxStripNo = sc.m_stripNo;
      firstIt=false;          
    } 
    
    if(phi < minPhi){
      minPhi=phi;
    }
    if(phi > maxPhi){
      maxPhi=phi;
    }
    
    if (maxStripNo<sc.m_stripNo)
      maxStripNo=sc.m_stripNo;
    
    m_physStripsInRingFromRolls++;
  }// loop end
  
}
//#############################################################################
/**
 *
 * \brief Filters strip
 *
 */
//#############################################################################
void RPCRingFromRolls::filterMixedStrips(){

  if (m_didFiltering){ // run once
    return;
  }
  
  m_didFiltering=true;
  
  if (m_region != 0 || m_hwPlane != 4) 
    return;
  
//  std::cout << "Another filtering" << std::endl;
  
  RPCRingFromRolls::phiMapCompare compare;
  /*
                  // compare(a,b) <=>  (a<b)
  if (compare(lowestPhiMap[scTemp.m_detRawId],it->first))
  {
    lowestPhiMap[scTemp.m_detRawId]=it->first;
  }
  */
  /*  
  std::map<uint32_t,float> phiCutMap;
  //Iterate over the chambers. For each chamber calculate a cut in phi
  RPCDetInfoPhiMap::const_iterator it = m_RPCDetPhiMap.begin();
  for(;it!=m_RPCDetPhiMap.end();it++){
  
  
}*/
  
  // filter strips
  
  uint32_t curRawID = 0, firstRawID = 0;
  bool firstRun=true;
  float phiCut = 0;
  
  std::vector<uint32_t> procChambers; // Stores rawIds of chambers that were processed
    
  GlobalStripPhiMap::iterator it = m_stripPhiMap.begin();
  for(;it!=m_stripPhiMap.end();it++){
    
    if(firstRun){
      firstRun=false;
      curRawID = it->second.m_detRawId;
      phiCut = m_RPCDetInfoMap[curRawID].getMaxPhi();
      firstRawID = curRawID; // First chamber is processed twice - at begin at the end of processing
    } 
    else {
      float phi = it->first;
      uint32_t rawID = it->second.m_detRawId;
  //    std::cout << rawID << " " << phi << " " << phiCut << std::endl;
      if (rawID!=curRawID){ // Region of mixed strips
        //if (procChambers.find(rawID)!=procChambers.end()){
        if (std::find(procChambers.begin(),procChambers.end(),rawID)!=procChambers.end()){
          if (rawID == firstRawID && procChambers.size() != 1) {} //do nothing for first processed chamber when  
                                                                  // proccesing it second time at the end
          else
            throw RPCException("The chamber should be allready processed");
        }
        if (compare(phi,phiCut)){  // compare(a,b) <=>  (a<b)
          //GlobalStripPhiMap::iterator ittemp = it;
          m_stripPhiMap.erase(it++);// delete strip pointed by it (not by it++ !)
          //ittemp--; // 
          //it=ittemp; 
          it--; // go to prev. element - loop will inc. it for us
        } 
        else { // Strip is ok - start new chamber
          procChambers.push_back(curRawID); // Store info, that the chamber was proccessd
          curRawID=rawID; // save new chamber id
          phiCut = m_RPCDetInfoMap[curRawID].getMaxPhi(); // get new cut
        }
      
      }
        
        
    }
  
  
  }
  
  

}
//#############################################################################
/**
 *
 * \brief Fills strip map with virtual strips
 * \bug RingFromRolls 4102 has diffrent no. of virtuals than 4002.
 * \todo Improve this function. Some curls seem to have wrong number of virtuals
 * \todo Possible xcheck - check if virtual and physical strips sum to 1152 or more - only for ref curl
 * \todo Implement check if we have symetry (x1xx vs x0xx; whell +y vs -y)
 *
 */
//#############################################################################
void RPCRingFromRolls::doVirtualStrips(){
  
  if (m_didVirtuals){ // run once
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

    int stripsToAdd = (int)std::floor(delta/dphi)-1;
    if ( isRefPlane() && m_hwPlane==6)
       stripsToAdd++; 
    
    double dphi1 = dphi;
    if (m_fixRPCGeo){
      if ( m_curlId == 6105 || m_curlId == 6106 || m_curlId == 6107 ){
         stripsToAdd+=5; 
         dphi1 = delta/(stripsToAdd+1);
      }
    }

    stripCords sc;
    sc.m_detRawId = rawDetIDLast;
    sc.m_stripNo = 0;
    sc.m_isVirtual = true;
    for (int i = 0;i<stripsToAdd;i++){
        sc.m_stripNo--;
        newVirtualStrips[phiMaxLast+dphi1*(i+1)]=sc;
        m_virtStripsInRingFromRolls++;
    }
  } // loop over dets end 
  
  
  if ( (isRefPlane()) && (m_virtStripsInRingFromRolls+m_physStripsInRingFromRolls!=1152)){
    
    edm::LogError("RPCTrigger")<<"Trouble. Reference curl " << getRingFromRollsId() 
        << " has " << m_virtStripsInRingFromRolls+m_physStripsInRingFromRolls << " strips."
        << " (v=" << m_virtStripsInRingFromRolls 
        << ";p=" << m_physStripsInRingFromRolls <<")";         
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
void RPCRingFromRolls::setRefPlane() {
  
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
bool RPCRingFromRolls::isRefPlane() const { return m_isRefPlane;} ///< Returns value of m_isReferencePlane
int RPCRingFromRolls::getMinTower() const { return m_towerMin;} ///< Returns value of min m_tower
int RPCRingFromRolls::getMaxTower() const{ return m_towerMax;} ///< Returns value of max m_tower
int RPCRingFromRolls::getRingFromRollsId() const{ return m_curlId;} ///< Returns value of max m_tower


//#############################################################################
//##
//## \note RingFromRolls of hwPlane = 2, roll = 8 is connected nowhere. Why?
//##
//#############################################################################


// Straigth from ORCA
const int RPCRingFromRolls::m_mrtow [RPCRingFromRolls::IROLL_MAX+1] [RPCRingFromRolls::NHPLANES] [RPCRingFromRolls::NPOS] =
//const int RPCRingFromRolls::m_mrtow [] [] [] =
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

const int RPCRingFromRolls::m_mrlogp [RPCRingFromRolls::IROLL_MAX+1] [RPCRingFromRolls::NHPLANES] [RPCRingFromRolls::NPOS] =
//const int RPCRingFromRolls::m_mrlogp [] [] [] =
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
const unsigned int RPCRingFromRolls::m_LOGPLANE_SIZE[RPCRingFromRolls::TOWERMAX+1][RPCRingFromRolls::NHPLANES] = {
//const unsigned int RPCRingFromRolls::m_LOGPLANE_SIZE[17][6] = {
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
* \brief prints the contents of a RPCRingFromRolls.
*
*/
//#############################################################################
void RPCRingFromRolls::printContents() {

  /*
  if (getRingFromRollsId()==71001){

    GlobalStripPhiMap::const_iterator it;
    for (it=m_stripPhiMap.begin(); it != m_stripPhiMap.end(); it++){
      LogDebug("RPCTrigger") << "phi" << it->first
          << " m_stripNo=" << (it->second).m_stripNo
          << " m_isVirtual=" << (it->second).m_isVirtual
    }
  }//*/

  if (isRefPlane())
    LogDebug("RPCTrigger") << " Reference plane:";
  else
    LogDebug("RPCTrigger") << " Normal plane:";
  
  LogDebug("RPCTrigger") << "No. of DetInfo's " << m_RPCDetInfoMap.size()
      << "; towers: min= " << m_towerMin 
      << " max= " << m_towerMax 
      << "|globRoll= " << m_globRoll
      << " hwPlane= " << m_hwPlane
      << "|strips:"
      << " phys= " << m_physStripsInRingFromRolls
      << " virt= " << m_virtStripsInRingFromRolls
      << " all= " << m_virtStripsInRingFromRolls+m_physStripsInRingFromRolls
      << "|strips conneced: " << m_links.size(); // with or without virtual strips. check it, it may have changed
  
  
  /*
  RPCDetInfoPhiMap::const_iterator it;
  for (it = m_RPCDetPhiMap.begin(); it != m_RPCDetPhiMap.end(); it++){
  
    //LogDebug("RPCTrigger")
    //    << "Phi: " << it->first << " "
    //   << "detId: " << it->second  <<
  
    
    m_RPCDetInfoMap[it->second].printContents();
        
  }
  //*/
  /*
  GlobalStripPhiMap::const_iterator it;
  for (it = m_stripPhiMap.begin(); it != m_stripPhiMap.end(); it++){
  LogDebug("RPCTrigger")
        << "Phi: " << it->first 
        << " detId: " << it->second.m_detRawId 
        << " m_stripNo: " << it->second.m_stripNo  << 
  }//*/

}


