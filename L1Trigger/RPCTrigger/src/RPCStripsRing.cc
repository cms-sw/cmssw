// -*- C++ -*-
//
// Package:     RPCConeBuilder
// Class  :     RPCStripsRing
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Tomasz Fruboes
//         Created:  Tue Feb 26 15:13:10 CET 2008
// $Id: RPCStripsRing.cc,v 1.3 2011/02/25 16:56:18 fruboes Exp $
//

// system include files

// user include files
//#include "L1TriggerConfig/RPCConeBuilder/interface/RPCStripsRing.h"
#include "L1Trigger/RPCTrigger/interface/RPCStripsRing.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

RPCStripsRing::RPCStripsRing() :
    m_hwPlane(-1),
    m_etaPartition(99),
    m_region(-2),
    m_isReferenceRing(false),
    m_didVirtuals(false),
    m_didFiltering(false)
{

}

RPCStripsRing::RPCStripsRing(const RPCRoll * roll,
                             boost::shared_ptr<L1RPCConeBuilder::TConMap > cmap) :
    m_didVirtuals(false),
    m_didFiltering(false),
    m_connectionsMap(cmap)
{
  
  RPCDetId detId = roll->id();
  RPCGeomServ grs(detId);
  
  m_etaPartition = grs.eta_partition();
  m_hwPlane = calculateHwPlane(roll);
  
  m_isReferenceRing = false;
  
  m_region = detId.region();
  
  int ring = detId.ring();
  
  if (m_region == 0 && std::abs(ring)<2 && m_hwPlane == 2) // for barell wheel -1,0,1 refplane is hwPlane=2
      m_isReferenceRing = true;
  else if (m_region == 0 && std::abs(ring)==2 && m_hwPlane == 6) // for barell wheel -2,2 refplane is hwPlane=6
      m_isReferenceRing = true;
  else if (m_region != 0 && m_hwPlane == 2) // for endcaps
      m_isReferenceRing = true;
  
  
  if( getRingId() == 2008 || getRingId() == 2108) //exception: endcaps;hwplane 2;farest roll from beam
      m_isReferenceRing = false;
    
  addRoll(roll);
}


void RPCStripsRing::addRoll(const RPCRoll * roll){

//  RPCDetId detId = roll->id();
  
  if (getRingId() !=  getRingId(roll) ){
     throw cms::Exception("RPCInternal") << "RPCStripsRing::addRoll ringsIds dont match \n";
  }
  
  //iterate over the strips of this roll
  for (int i=1; i<=roll->nstrips(); i++ ) { 
    
       LocalPoint lStripCentre = roll->centreOfStrip(i);
       GlobalPoint gStripCentre = roll->toGlobal(lStripCentre);
       float phiRaw = gStripCentre.phi();
       
       TStrip newStrip(roll->id().rawId(), i);
       (*this)[phiRaw] = newStrip;
       
  }

}

int RPCStripsRing::getRingId(int etaPart, int hwPlane){

  int sign = 1; // positive
  if (etaPart < 0){
    sign = 0;
  }
   
  return  1000*(hwPlane) +     //1...6
          100*( sign ) + //
          1*( std::abs(etaPart) );     //-17...17
  
}

int RPCStripsRing::getRingId(){

  return getRingId(m_etaPartition, m_hwPlane);
   
}

int RPCStripsRing::getRingId(const RPCRoll * roll) {

      
   RPCDetId detId = roll->id();
   RPCGeomServ grs(detId);
   int etaPartition = grs.eta_partition();
   int hwPlane = calculateHwPlane(roll);
   
   return getRingId(etaPartition, hwPlane);
      
}


//  hwPlane is  station number for endcaps
//  for barrell numbering goes 1 5 2 6 3 4 (first number means plane closest to the beam)
int RPCStripsRing::calculateHwPlane(const RPCRoll * roll){

  int hwPlane = -1;  
  RPCDetId detId = roll->id();
  int station = detId.station();
  int layer = detId.layer(); 
  int region = detId.region(); 
  
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
  
  /*if (hwPlane < 1)
    std::cout << "prb: " << hwPlane << " "
        << region << " "
        << station << " "
  << layer << std::endl;*/
  if (hwPlane < 0) {
      throw cms::Exception("RPCInternal") << "Calculated negative hwplane \n";
  }
  
  
  return hwPlane;
  
}

void RPCStripsRing::filterOverlapingChambers(){
    
  if(m_didFiltering) return;
  m_didFiltering = true;
  
  if (m_region != 0 || m_hwPlane != 4) 
     return;
  
  typedef std::map<uint32_t,int> TDetId2StripNo;
  TDetId2StripNo det2stripNo;
  
  // Note: we begin in middle of first chamber (ch1), we have to handle that
  int ch1BegStrips = 0; // no of strips on the begining of the map (first=last chamber of map)
  int ch1EndStrips = 0; // no of strips on the end of the map (first=last chamber of map)
  
  // How many strips has each chamber?
  RPCStripsRing::iterator it = this->begin();
  uint32_t ch1Det = it->second.m_detRawId;
  for (; it!=this->end(); ++it){
    
    if ( det2stripNo.find(it->second.m_detRawId) == det2stripNo.end()){
      det2stripNo[it->second.m_detRawId]=1;      // Add new chamber to a map, set strip cnt to 1
    } else {
      ++det2stripNo[it->second.m_detRawId];     // Increase strip count of a chamber
    }
    
    if (det2stripNo.size() == 1 && ch1Det == it->second.m_detRawId) {
      ++ch1BegStrips;
    } else if (ch1Det == it->second.m_detRawId){
      ++ch1EndStrips;
    }
    
  }
  
  det2stripNo[ch1Det]-=ch1EndStrips;
  
 // std::cout << ch1BegStrips << " " << ch1EndStrips << std::endl;
  
  //TDetId2StripNo::iterator itIds = det2stripNo.begin();
  //for(;itIds!=det2stripNo.end();++itIds){
//    std::cout << itIds->first << " " << itIds->second << std::endl;
//  }
  
  
  
  it = this->begin();
  uint32_t lastDet = it->second.m_detRawId;
  while ( it!=this->end() ){
    
    if (det2stripNo[it->second.m_detRawId] < 0) {
      throw cms::Exception("RPCInternal") << " RPCStripsRing::filterOverlapingChambers() - no strips left \n";
    }
    if ( it->second.m_detRawId == lastDet) {
      --det2stripNo[lastDet];
      ++it;
    } else if (det2stripNo[lastDet] == 0) { // no more strips left in lastDet, proceed to new det
      
      if (lastDet == ch1Det) {
        det2stripNo[ch1Det]+=ch1EndStrips;
      }
      
      lastDet = it->second.m_detRawId;
      --det2stripNo[lastDet];
      ++it;
    } else { // there are still strips in last det, delete current strip
      --det2stripNo[it->second.m_detRawId];
      RPCStripsRing::iterator itErase = it;
      ++it;
      //std::cout << "Removing strip " <<  it->second.m_detRawId << " " << (int)it->second.m_strip << std::endl;
      this->erase(itErase); 
    }
    
  }
  
  

}

void RPCStripsRing::fillWithVirtualStrips()
{
  

  if(m_didVirtuals) return;
  m_didVirtuals = true;

  const float pi = 3.141592654;
  double dphi=2.0*pi/1152; // defines angular granulation of strips.
  
  RPCStripsRing stripsToInsert;
    
  
  float delta = 0;
  int stripsToAdd = 0;
  
  
  RPCStripsRing::iterator it = this->begin();
  RPCStripsRing::iterator itLast = this->begin();
  for (; it!=this->end(); ++it){
  
    /*std::cout << it->first << " "
        << it->second.m_detRawId << " "
        << (int)it->second.m_strip << std::endl;
    */
    
    delta = it->first - itLast->first;        
    if (it == itLast || // skip first loop iteration
        itLast->second.m_detRawId == it->second.m_detRawId || // insert strips between two chambers only
        delta < 0)
    {
      itLast = it;
      continue;
    }
    
    
    stripsToAdd = (int)std::floor(delta/dphi)-1;
    //std::cout << delta << " " << stripsToAdd << std::endl;
    
    if ( isReferenceRing() && m_hwPlane==6) ++stripsToAdd;
    
    for (int i = 0;i<stripsToAdd;++i){
      
      stripsToInsert[itLast->first+dphi*(i+1)]=TStrip();
    
    }
    
    itLast = it; 
  }
  // TODO: check delta between first and last strip in map
  
  this->insert(stripsToInsert.begin(),stripsToInsert.end());

  
  
}
void RPCStripsRing::createRefConnections(TOtherConnStructVec & otherRings, int logplane, int logplaneSize)
{
  //*
   /*std::cout << "RefCon for " << getRingId() 
       << " (" << getEtaPartition()<<  ")"
       << " tower: " << getTowerForRefRing()
       << " ; connected: "
       << otherRings.size() 
       << std::endl
       << std::endl;    
  //*/
       
  // XXX - TODO: warning on wrong logplaneSize
  
   if(!this->isReferenceRing()){
      throw cms::Exception("RPCInternal") << " RPCStripsRing::createRefConnections "
         << " called for non-reference ring \n";
   }
   
   /*
   if (logplaneSize!=8) {
     throw cms::Exception("RPCInternal") << " RPCStripsRing::createRefConnections "
         << " called for lpSize " << logplaneSize << " \n";
     
   }*/
   const float pi = 3.141592654;
   const float offset = (5./360.)*2*pi; // XXX
   
   //find first reference strip of first PAC (the strip with phi ~= 5deg)
   RPCStripsRing::iterator starEndIt = this->begin();
   while ( (++starEndIt)->first < offset ); 
         
   RPCStripsRing::iterator it = starEndIt;
   //--starEndIt;
   
   float angle = 0;
   int curPACno = -1;
   int curStripNo = 0;
   int curBegStripNo=0;
   
  bool firstIter = true;

   while(it!=starEndIt || firstIter ) { // iterate over strips
 

     firstIter = false;
      // New PAC  
     if(curStripNo%logplaneSize==0){ 
         ++curPACno; 
         curBegStripNo=curStripNo;
         RPCStripsRing::iterator plus8 = it;
         bool skipOccured = false;
         for (int i=0;i<7;++i){  
            ++plus8;
            if (plus8==this->end()){
               plus8=this->begin();
               skipOccured = true;
            }
         }
         
         // calculate angle
         float phi= it->first;
         float phiP8= plus8->first;
         if (skipOccured){
            // phiP8 is negative
            // phi is positive
            // xcheck
           if (phi*phiP8 > 0){
             throw cms::Exception("RPCInternal") << " RPCStripsRing::createRefConnections phi/phi8 error \n";
           }
           angle = (2*pi+phiP8+phi)/2;
           if(angle > pi){ // should land on positive side
              angle -= 2*pi;
           } 
            
           if (std::abs(angle) > pi) {
               throw cms::Exception("RPCInternal") << " RPCStripsRing::createRefConnections "
                     << " problem with angle calc \n";
           }
         }
         else {
           angle = (phiP8+phi)/2;
         }
         //std::cout << curPACno << " " << phiP8 << " " << phi << " "  << angle << std::endl;
         
         
         TOtherConnStructVec::iterator itOt = otherRings.begin();
         for (;itOt!=otherRings.end();++itOt){
           itOt->m_it->second.createOtherConnections(getTowerForRefRing(),
                                                     curPACno, 
                                                     itOt->m_logplane,
                                                     itOt->m_logplaneSize,
                                                     angle);
         }
      }
      
      
      if ( !it->second.isVirtual() ){
        L1RPCConeBuilder::TStripCon newCon;
        newCon.m_tower = getTowerForRefRing();
        newCon.m_PAC = curPACno;
        newCon.m_logplane = logplane;
        newCon.m_logstrip=curStripNo-curBegStripNo;
        //std::cout << " Adding con for " << it->second.m_detRawId << std::endl;
        (*m_connectionsMap)[it->second.m_detRawId][it->second.m_strip].push_back(newCon);
        //std::cout << " Adding ref connection " << std::endl;
      }
      ++curStripNo;
      ++it;
      if (it==this->end()){
         it=this->begin();
      }
       
   } // iteration over strips ends
   
   //std::cout << " refcon: " << curPACno << " PACs" << std::endl;
   //std::cout << "After refCon: " << m_connectionsMap.size() << std::endl;

}

void RPCStripsRing::createOtherConnections(int tower, int PACno, int logplane, int logplaneSize, float angle) {

   //std::cout << "    OtherCon for " << getRingId() << std::endl;

   if(this->isReferenceRing()){
      throw cms::Exception("RPCInternal") << " RPCStripsRing::createOtherConnections "
            << " called for reference ring \n";
   }


   RPCStripsRing::const_iterator it = this->lower_bound(angle);
   
   
   if (it == this->end())
     it = this->begin();
   
   for (int i=0; i < logplaneSize/2; i++){ 
     
      if (it==this->begin())
        it=this->end();  // (m_stripPhiMap.end()--) is ok.
      
      --it;
   }
  
     
   for (int i=0; i < logplaneSize; i++){
    
     if (! it->second.isVirtual() ){
        L1RPCConeBuilder::TStripCon newCon;
        newCon.m_tower = tower;
        newCon.m_PAC = PACno;
        newCon.m_logplane = logplane;
        newCon.m_logstrip= i;
        (*m_connectionsMap)[it->second.m_detRawId][it->second.m_strip].push_back(newCon);
        //std::cout << " Adding other connection " << std::endl;
      }
  
      ++it;
      if (it==this->end())
        it=this->begin();
   }
     
}

// Defines to which tower this ring (only ref ring) belongs
int RPCStripsRing::getTowerForRefRing(){

  int ret = 0;
  
  if(!this->isReferenceRing()){
    throw cms::Exception("RPCInternal") << " RPCStripsRing::getTowerForRefRing() "
        << " called for non reference ring \n";
  }

  int etaAbs = std::abs(getEtaPartition());
  if (etaAbs < 8) {
    ret = getEtaPartition();
  } else if (etaAbs > 8) {
    int sign = (getEtaPartition() > 0 ? 1 : -1);
    ret = getEtaPartition()-sign;
  } else {
    throw cms::Exception("RPCInternal") << " RPCStripsRing::getTowerForRefRing() "
        << " called for etaPartition 8 \n";
  }



  return ret;

}
/*
      struct TStripCon{
        signed char m_tower;
        unsigned char m_PAC;
        unsigned char m_logplane;
        unsigned char m_logstrip;
      };
      typedef std::vector<TStripCon> TStripConVec;
      typedef std::map<unsigned char, TStripConVec> TStrip2ConVec;
      typedef std::map<uint32_t, TStrip2ConVec> TConMap;

      // compressed connections
      struct TCompressedCon{
        signed char m_tower;
        unsigned char m_PAC;
        signed char m_offset;
        signed char m_mul;
      };
      typedef std::vector<TCompressedCon> TCompressedConVec;
      typedef std::map<uint32_t, TCompressedConVec> TCompressedConMap;

*/


void RPCStripsRing::compressConnections(){

  
  L1RPCConeBuilder::TConMap::iterator itChamber = m_connectionsMap->begin();
  
  boost::shared_ptr<L1RPCConeBuilder::TConMap > uncompressedConsLeft
  = boost::shared_ptr<L1RPCConeBuilder::TConMap >(new L1RPCConeBuilder::TConMap());
  
  m_compressedConnectionMap =        
               boost::shared_ptr<L1RPCConeBuilder::TCompressedConMap >
               (new L1RPCConeBuilder::TCompressedConMap());
  
  
  int compressedCons = 0, uncompressedConsBefore = 0, uncompressedConsAfter = 0;
  
//   int offsetMin =0, offsetMax =0;
  
  for( ;itChamber!=m_connectionsMap->end(); ++itChamber ){
    
    uint32_t detId = itChamber->first;
    
    for (L1RPCConeBuilder::TStrip2ConVec::iterator itStrip = itChamber->second.begin();
         itStrip!=itChamber->second.end();
         ++itStrip)
    {
      
      // Iterate over strip Connections
      for(L1RPCConeBuilder::TStripConVec::iterator itConn = itStrip->second.begin();
          itConn!=itStrip->second.end(); 
          ++itConn)
      {
        // Check if this connection isn't allready present in the compressed map 
        ++uncompressedConsBefore;
        bool alreadyDone=false; 
        if (m_compressedConnectionMap->find(detId)!=m_compressedConnectionMap->end()){
          
          // iterate over the vec, check element by element
          for(L1RPCConeBuilder::TCompressedConVec::iterator itCompConn=(*m_compressedConnectionMap)[detId].begin();
              itCompConn!=(*m_compressedConnectionMap)[detId].end();
              ++itCompConn)
          {
            if (itCompConn->m_tower ==  itConn->m_tower
                && itCompConn->m_PAC ==  itConn->m_PAC
                && itCompConn->m_logplane ==  itConn->m_logplane) // connection allready compressed 
            {
              alreadyDone=true;
              
              int logStrip = itCompConn->m_mul*itStrip->first+itCompConn->m_offset;
              if (logStrip != itConn->m_logstrip){
                //copy the problematic connection to the "safe" map
                (*uncompressedConsLeft)[detId][itStrip->first].push_back(*itConn);
                ++uncompressedConsAfter;
                edm::LogWarning("RPCTriggerConfig") << " Compression failed for det " << detId 
                  << " strip " << (int)itStrip->first
                  << " . Got " << (int)logStrip
                  << " expected " << (int)itConn->m_logstrip
                  << std::endl;
              } else {
                itCompConn->addStrip(itStrip->first);
              }
              
            }
          } // compressed connection iteration end
        }  
        //if (detId==637569977) std::cout << " Buld cons for strip " << (int)itStrip->first << std::endl;
        
        
        if (!alreadyDone){
            // find another strip contributing to the same PAC,tower,logplane
          L1RPCConeBuilder::TStrip2ConVec::iterator itStripOther = itStrip;  
          ++itStripOther;
          bool otherStripFound = false;
          signed char mul = 1;
          for (;itStripOther!=itChamber->second.end() && !otherStripFound;
               ++itStripOther)
          {
            for(L1RPCConeBuilder::TStripConVec::iterator itConnOther = itStripOther->second.begin();
                itConnOther!=itStripOther->second.end(); 
                ++itConnOther)
            {
              if (itConnOther->m_tower ==  itConn->m_tower
                  && itConnOther->m_PAC ==  itConn->m_PAC
                  && itConnOther->m_logplane ==  itConn->m_logplane) // connection to same PAC,logplane
              {
                otherStripFound = true;
                if ( (itStripOther->first-itStrip->first)*(itConnOther->m_logstrip-itConn->m_logstrip) < 0 ){
                  mul = -1;                
                } 
                break;
              }
            } // otherConnections iter ends
          } // otherStrip iter ends
          
          /*
          if (itConn->m_tower==3 && itConn->m_PAC==73 && itConn->m_logplane==4 && detId==637569977){
            std::cout << " Buld cons for strip " << (int)itStrip->first;
            if (otherStripFound)
              std::cout << " other strip " << itStrip->first;
            else 
              std::cout << " no other strip ";
            
            std::cout << std::endl;
            
        }*/
          
          L1RPCConeBuilder::TCompressedCon nCompConn;
          nCompConn.m_tower = itConn->m_tower;
          nCompConn.m_PAC   = itConn->m_PAC;
          nCompConn.m_logplane   = itConn->m_logplane;
          nCompConn.m_mul  = mul;
          nCompConn.m_offset  = itConn->m_logstrip - mul*(signed short)(itStrip->first);
          nCompConn.addStrip(itStrip->first);
          
          if (otherStripFound){
            
          }  else { 
            
            //  uncompressedConsLeft[detId][itStrip->first].push_back(*itConn);
            //  ++uncompressedConsAfter;
          }
          (*m_compressedConnectionMap)[detId].push_back(nCompConn);
          ++compressedCons;


        } // if(!allreadyDone)
      }// iterate on connections
    }// iterate on strips
  } // iterate on chambers
   
  // 159 -87
  //std::cout << offsetMax << " TT " << offsetMin << std::endl;
  
  edm::LogInfo("RPCTriggerConfig") 
      << " Compressed: " << compressedCons<< " " << sizeof(L1RPCConeBuilder::TCompressedCon)
      << " Uncompressed before: " << uncompressedConsBefore<< " " << sizeof(L1RPCConeBuilder::TStripCon)
      << " Uncompressed after: " << uncompressedConsAfter << " " << sizeof(L1RPCConeBuilder::TStripCon);
  m_connectionsMap = uncompressedConsLeft;

}

