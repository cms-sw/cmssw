#ifndef L1Trigger_RPCConeBuilder_RPCStripsRing_h
#define L1Trigger_RPCConeBuilder_RPCStripsRing_h
// -*- C++ -*-
//
// Package:     RPCConeBuilder
// Class  :     RPCStripsRing
// 
/**\class RPCStripsRing RPCStripsRing.h L1Trigger/RPCTrigger/interface/RPCStripsRing.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Tomasz Fruboes
//         Created:  Tue Feb 26 15:13:17 CET 2008
// $Id: RPCStripsRing.h,v 1.1 2009/06/01 13:58:16 fruboes Exp $
//

#include <map>
#include <vector>
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"

class RPCRoll;


// XXX TODO: move into namespace?
struct TStrip {
  TStrip() : m_detRawId(0), m_strip(0) {};
  TStrip(int rawId, int stripNo) : m_detRawId(rawId), m_strip(stripNo) {};
  bool isVirtual() const {return  m_detRawId==0;};
  uint32_t m_detRawId;
  unsigned char m_strip;
};


class RPCStripsRing : public std::map<float, TStrip >
{

   public:
      //                | ringId
      typedef std::map<int, RPCStripsRing> TIdToRindMap;
      
      struct TOtherConnStruct {
         
         TOtherConnStruct() : m_logplane(0), m_logplaneSize(0), m_it(0) {};
         short m_logplane;
         short m_logplaneSize;
         TIdToRindMap::iterator m_it;
      };
      
      typedef std::vector<TOtherConnStruct> TOtherConnStructVec;
      
            
      RPCStripsRing(const RPCRoll * roll, 
                    boost::shared_ptr<L1RPCConeBuilder::TConMap > cmap);
                    
      RPCStripsRing();
      virtual ~RPCStripsRing() {};
      
      void addRoll(const RPCRoll * roll);
      
      // RPCDetInfo::getRingFromRollsId()
      static int getRingId(int etaPart, int hwPlane);
      int getRingId();   /// Calculate ringId for this ring
      static int getRingId(const RPCRoll * roll); /// Calculate ringId for any given RPCRoll
      

      static int calculateHwPlane(const RPCRoll * roll);
            
      void filterOverlapingChambers();
      void fillWithVirtualStrips();

      void createRefConnections(TOtherConnStructVec & otherRings, int logplane, int logplaneSize);
      void createOtherConnections(int tower, int PACno, int logplane, int logplanesize, float angle);
      
      
      int getHwPlane() {return m_hwPlane;};
      
      int getEtaPartition() {return m_etaPartition;};
      bool isReferenceRing(){return m_isReferenceRing;};
      int getTowerForRefRing();
      
      void compressConnections();
      boost::shared_ptr<L1RPCConeBuilder::TConMap > getConnectionsMap() 
              { return m_connectionsMap;};
              
      boost::shared_ptr<L1RPCConeBuilder::TCompressedConMap> getCompressedConnectionsMap() 
      { 
        return m_compressedConnectionMap;
      };
      
  private:
    
      
      int m_hwPlane;
      int m_etaPartition; // m_globRoll previously
      int m_region; 

      bool m_isReferenceRing; // m_isRefPlane previously
      bool m_didVirtuals; // m_isRefPlane previously
      bool m_didFiltering;    
      
      boost::shared_ptr<L1RPCConeBuilder::TConMap > m_connectionsMap;  
      boost::shared_ptr<L1RPCConeBuilder::TCompressedConMap > m_compressedConnectionMap;
};


#endif
